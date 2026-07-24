# SPDX-FileCopyrightText: 2026 Fan Liu
# SPDX-License-Identifier: MIT
#
# SATCP arc-length force-control solver for legacy FEniCS/DOLFIN.
#
# Associated manuscript:
#   Fan Liu and Q. L. Duan,
#   "A secant--approximate-tangent curvature predictor for arc-length
#   tracing of geometrically nonlinear postbuckling paths".
#
# The solver uses the public Python interface of legacy FEniCS/DOLFIN.
# See README.md for dependencies, citation information, benchmark settings,
# and limitations.
#
# Released under the MIT License. See LICENSES/MIT.txt.

"""SATCP arc-length solver for legacy FEniCS/DOLFIN.

The public ``force_control`` class performs one attempted arc-length increment
per call to :meth:`solve`. It supports both assembled vector loads, including
``PointSource`` loads, and UFL external-load forms.

The code is intended for research reproducibility. Users remain responsible
for checking discretisation, boundary conditions, convergence tolerances, and
solution-path accuracy for their own problems.
"""

from dolfin import *
from math import acos, cos, isfinite, pi, sin, sqrt

try:
    DOLFIN_EPS
except Exception:
    DOLFIN_EPS = None


class force_control:
    """
    FEniCS/DOLFIN arc-length force-control solver using the
    secant--approximate-tangent curvature predictor (SATCP).

    This implementation follows the algorithm described in the accompanying paper:

    1. The first accepted increment is solved as a prescribed-load standard
       Newton step.  In the split equations this is achieved by A=0, a=0,
       and b=1, so the load-factor correction is zero.
    2. During start-up, Kadapa's simple extrapolation predictor (SEP) is used
       only when the required accepted history is unavailable.
    3. The approximate tangent is constructed from the unit-load response
       used in the last *actual* Newton correction of the latest accepted
       increment.  No unit-load response is recomputed after convergence.
    4. The current effective curvature is estimated from the angle between
       the latest accepted secant and the approximate endpoint tangent.
    5. The next secant rotation is

           phi = 1/2*kappa_n*Ds_next
               + 1/6*kappa_s,n*Ds_next**2,

       followed by the non-negative truncation and ``phi_max`` clipping.
    6. Curvature and unit-load-response trial states are committed only after
       an attempted increment converges; failed attempts do not contaminate
       accepted history.
    7. Under the adaptive arc-length rule used in the accompanying study,
       Ds is doubled after consecutive converged attempts and capped by the
       first accepted arc length; the first failed attempt halves Ds and
       consecutive failed attempts quarter Ds.

    Parameters
    ----------
    psi : float
        Arc-length metric parameter.  psi=1 gives the spherical constraint.
    lmbda0 : float
        Prescribed load-factor increment used for the first accepted step.
        It also supplies the initial trial value of the arc length.
    max_iter : int
        Maximum number of Newton iterations per attempted increment.
    u : dolfin.Function
        State function containing the primary finite element unknowns.
    F_int : UFL form
        Internal-force/residual form before subtraction of the external load.
    F_ext : UFL form or assembled dolfin.Vector
        Reference external load.  A Vector is suitable for PointSource loads.
    bcs : sequence
        Dirichlet boundary conditions.
    J : UFL form
        Tangent/Jacobian form.
    load_factor : optional mutable Constant-like object
        Required when ``F_ext`` is a UFL form containing the load factor.
    abs_tol : float, optional
        Absolute convergence tolerance for sqrt(||R||^2 + A^2).
        The default is 1e-6.
    rel_tol : optional
        Retained for compatibility with earlier solver interfaces.  It is not
        used because this implementation uses only the absolute combined norm.
    solver : str, optional
        DOLFIN linear solver selector passed to ``solve``.
    norm_tol, angle_tol : float, optional
        Numerical safeguards for normalization and angle construction.
    phi_max : float, optional
        Maximum additional tangent-to-next-secant rotation.  Default 20 deg.
    num_accepted_increments_before_curvature : int, optional
        Minimum number of accepted increments required before the curvature
        predictor is activated.  The SATCP implementation uses 1.
    denominator_tol : float, optional
        Safeguard for the scalar split-equation denominator.
    """

    def __init__(
        self,
        psi,
        lmbda0,
        max_iter,
        u,
        F_int,
        F_ext,
        bcs,
        J,
        load_factor=None,
        abs_tol=1.0e-6,
        rel_tol=DOLFIN_EPS,
        solver="default",
        norm_tol=1.0e-12,
        angle_tol=1.0e-8,
        phi_max=20.0 * pi / 180.0,
        num_accepted_increments_before_curvature=1,
        denominator_tol=1.0e-14,
    ):
        if lmbda0 <= 0.0:
            raise ValueError("lmbda0 must be a positive first-step load increment.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if num_accepted_increments_before_curvature < 1:
            raise ValueError(
                "num_accepted_increments_before_curvature must be at least 1."
            )

        self.psi = float(psi)
        self.abs_tol = float(abs_tol)
        self.rel_tol = rel_tol  # API compatibility; not used in convergence.
        self.max_iter = int(max_iter)
        self.norm_tol = float(norm_tol)
        self.angle_tol = float(angle_tol)
        self.phi_max = float(phi_max)
        self.denominator_tol = float(denominator_tol)
        self.num_accepted_increments_before_curvature = int(
            num_accepted_increments_before_curvature
        )

        self.u = u
        self.F_int = F_int
        self.F_ext = F_ext
        self.bcs = bcs
        self.J = J
        self.load_factor = load_factor
        self.solver = solver

        self.vector_load = hasattr(F_ext, "axpy") and hasattr(F_ext, "inner")
        if self.vector_load:
            self.residual = F_int
        else:
            if self.load_factor is None:
                raise ValueError(
                    "F_ext is a UFL form, so load_factor must be provided. "
                    "For a point load, pass an assembled reference Vector."
                )
            self.residual = F_int - F_ext

        # Accepted equilibrium histories. The solver starts from the zero
        # equilibrium state and treats lmbda0 as the first load increment.
        initial_u = self.u.vector().copy()
        self.u_n = initial_u.copy()
        self.u_n_1 = initial_u.copy()
        self.lmbda_n = 0.0
        self.lmbda_n_1 = 0.0

        self.initial_load_increment = float(lmbda0)
        self.delta_s = float(lmbda0)
        self.delta_s_n = float(lmbda0)

        # Before the first accepted step, all three arc-length values are
        # initialised from lmbda0 and reset after the first accepted step.
        self.delta_s_max = float(lmbda0)
        self.delta_s_min = float(lmbda0)

        self.lmbda = 0.0
        self.__set_load_factor(self.lmbda)

        # counter = number of accepted increments.  Each solve() call performs
        # exactly one attempted increment; counter changes only on acceptance.
        self.counter = 0
        self.attempted_steps = 0
        self.converged = False
        self.converged_prev = False
        self.restart_count = 0

        # Iteration counters record actual Newton corrections only.  The
        # final convergence check is not counted as an iteration.  The
        # correction counters are retained as compatibility aliases.
        self.last_iters = 0
        self.total_iters = 0
        self.last_corrections = 0
        self.total_corrections = 0

        self.delta_s0 = None
        self.accepted_arc_length_unit = None
        self.last_accepted_arc_length = 0.0
        self.accepted_arc_length = 0.0
        self.accepted_arc_length_by_delta_s0 = 0.0
        self.accepted_arc_length_normalized = 0.0
        self.accepted_arc_length_steps = 0
        self.accepted_arc_lengths = [0.0]
        self.accepted_arc_lengths_by_delta_s0 = [0.0]
        self.accepted_arc_lengths_normalized = self.accepted_arc_lengths_by_delta_s0

        # Last accepted unit-load response used in the last actual correction.
        self.du_1_last = None
        self.has_du_1_last = False

        # Curvature state, committed only after a successful attempted step.
        self.kappa_prev = 0.0
        self.has_kappa_prev = False

        # Latest accepted augmented increment and unit direction.
        self.delta_u_n = self.__zero_like(initial_u)
        self.delta_lmbda_n = 0.0
        self.delta_u_n_bar = self.__zero_like(initial_u)
        self.delta_lmbda_n_bar = 0.0
        self.has_delta_u_n = False

        self.last_predictor_info = None
        self.__build_Fext_vec()

    # ------------------------------------------------------------------
    # Basic vector, load, and assembly utilities
    # ------------------------------------------------------------------
    def __update_nodal_values(self, u_new):
        self.u.vector().set_local(u_new.get_local())
        self.u.vector().apply("insert")

    def __set_load_factor(self, value):
        if self.load_factor is not None:
            self.load_factor.t = value

    @staticmethod
    def __zero_like(vec):
        out = vec.copy()
        out.zero()
        return out

    @staticmethod
    def __clamp(value, lower_bound, upper_bound):
        return min(upper_bound, max(lower_bound, value))

    def __build_Fext_vec(self):
        if self.vector_load:
            self.F_ext_vec = self.F_ext.copy()
        else:
            old_lmbda = self.lmbda
            self.__set_load_factor(1.0)
            self.F_ext_vec = assemble(self.F_ext)
            self.__set_load_factor(old_lmbda)

        for bc in self.bcs:
            bc.apply(self.F_ext_vec)

        self.FF = self.F_ext_vec.inner(self.F_ext_vec)
        if not isfinite(self.FF) or self.FF <= self.norm_tol ** 2:
            raise ValueError("The reference external-load vector has zero/invalid norm.")

    def __assemble_system_current(self):
        if self.vector_load:
            K, R = assemble_system(self.J, self.F_int, self.bcs)
            R.axpy(-self.lmbda, self.F_ext_vec)
            return K, R

        self.__set_load_factor(self.lmbda)
        return assemble_system(self.J, self.residual, self.bcs)

    # ------------------------------------------------------------------
    # Arc-length metric and predictor geometry
    # ------------------------------------------------------------------
    def __arc_inner(self, u1, lmbda1, u2, lmbda2):
        return u1.inner(u2) + self.psi * lmbda1 * lmbda2 * self.FF

    def __standardize_increment(self, u_inc, lmbda_inc):
        norm_squared = self.__arc_inner(
            u_inc, lmbda_inc, u_inc, lmbda_inc
        )
        valid = isfinite(norm_squared) and norm_squared > self.norm_tol ** 2

        if not valid:
            return self.__zero_like(u_inc), 0.0, False, 0.0

        norm_value = sqrt(norm_squared)
        return (
            u_inc / norm_value,
            lmbda_inc / norm_value,
            True,
            norm_value,
        )

    def __record_accepted_arc_length(self, ds_accepted):
        ds_accepted = float(ds_accepted)
        if ds_accepted <= 0.0:
            raise RuntimeError("Accepted arc length must be positive.")

        if self.delta_s0 is None:
            self.delta_s0 = ds_accepted
            self.accepted_arc_length_unit = ds_accepted

        self.last_accepted_arc_length = ds_accepted
        self.accepted_arc_length += ds_accepted
        self.accepted_arc_length_steps += 1
        self.accepted_arc_length_by_delta_s0 = (
            self.accepted_arc_length / self.delta_s0
        )
        self.accepted_arc_length_normalized = self.accepted_arc_length_by_delta_s0
        self.accepted_arc_lengths.append(self.accepted_arc_length)
        self.accepted_arc_lengths_by_delta_s0.append(
            self.accepted_arc_length_by_delta_s0
        )

    def get_accepted_arc_length(self, normalized=False):
        if normalized:
            return self.accepted_arc_length_by_delta_s0
        return self.accepted_arc_length

    def __compute_tangent_direction(self):
        # Reference tangent is [du_1_last; 1].  Orient it consistently with
        # the latest accepted secant under the augmented arc-length metric.
        orientation = self.__arc_inner(
            self.delta_u_n,
            self.delta_lmbda_n,
            self.du_1_last,
            1.0,
        )
        sign_value = 1.0 if orientation >= 0.0 else -1.0

        tangent_norm_squared = (
            self.du_1_last.inner(self.du_1_last) + self.psi * self.FF
        )
        valid = (
            isfinite(tangent_norm_squared)
            and tangent_norm_squared > self.norm_tol ** 2
        )
        if not valid:
            return self.__zero_like(self.du_1_last), 0.0, False

        tangent_norm = sqrt(tangent_norm_squared)
        return (
            sign_value * self.du_1_last / tangent_norm,
            sign_value / tangent_norm,
            True,
        )

    def __pure_curvature_direction(self, u_tan_bar, lmbda_tan_bar):
        cos_nt = self.__arc_inner(
            self.delta_u_n_bar,
            self.delta_lmbda_n_bar,
            u_tan_bar,
            lmbda_tan_bar,
        )
        cos_nt = self.__clamp(cos_nt, -1.0, 1.0)
        theta = acos(cos_nt)

        kappa_now = 2.0 * theta / max(self.delta_s_n, self.norm_tol)
        if self.has_kappa_prev:
            kappa_rate = (kappa_now - self.kappa_prev) / max(
                self.delta_s_n, self.norm_tol
            )
        else:
            kappa_rate = 0.0

        phi_target_raw = (
            0.5 * kappa_now * self.delta_s
            + (1.0 / 6.0) * kappa_rate * self.delta_s ** 2
        )
        phi_target = max(0.0, phi_target_raw)
        phi_clipped_by_max = False
        if isfinite(self.phi_max) and phi_target > self.phi_max:
            phi_target = self.phi_max
            phi_clipped_by_max = True

        # The local plane is undefined when q_n and q_T are effectively
        # parallel/antiparallel.  A non-positive predicted rotation also
        # degenerates to the approximate tangent direction.
        if (
            theta <= self.angle_tol
            or theta >= pi - self.angle_tol
            or phi_target <= self.angle_tol
        ):
            return {
                "u_pred_bar": u_tan_bar,
                "lmbda_pred_bar": lmbda_tan_bar,
                "kappa_now": kappa_now,
                "kappa_rate": kappa_rate,
                "phi_target": phi_target,
                "phi_target_raw": phi_target_raw,
                "phi_clipped_by_max": phi_clipped_by_max,
                "theta": theta,
                "valid": True,
                "used_tangent": True,
            }

        sin_theta = sin(theta)
        u_forward = (
            cos_nt * u_tan_bar - self.delta_u_n_bar
        ) / sin_theta
        lmbda_forward = (
            cos_nt * lmbda_tan_bar - self.delta_lmbda_n_bar
        ) / sin_theta

        (
            u_forward,
            lmbda_forward,
            forward_valid,
            _,
        ) = self.__standardize_increment(u_forward, lmbda_forward)

        if not forward_valid:
            return {
                "u_pred_bar": u_tan_bar,
                "lmbda_pred_bar": lmbda_tan_bar,
                "kappa_now": kappa_now,
                "kappa_rate": kappa_rate,
                "phi_target": phi_target,
                "phi_target_raw": phi_target_raw,
                "phi_clipped_by_max": phi_clipped_by_max,
                "theta": theta,
                "valid": True,
                "used_tangent": True,
            }

        u_pred = (
            cos(phi_target) * u_tan_bar
            + sin(phi_target) * u_forward
        )
        lmbda_pred = (
            cos(phi_target) * lmbda_tan_bar
            + sin(phi_target) * lmbda_forward
        )
        (
            u_pred_bar,
            lmbda_pred_bar,
            pred_valid,
            _,
        ) = self.__standardize_increment(u_pred, lmbda_pred)

        if not pred_valid:
            u_pred_bar = u_tan_bar
            lmbda_pred_bar = lmbda_tan_bar
            pred_valid = True
            used_tangent = True
        else:
            used_tangent = False

        return {
            "u_pred_bar": u_pred_bar,
            "lmbda_pred_bar": lmbda_pred_bar,
            "kappa_now": kappa_now,
            "kappa_rate": kappa_rate,
            "phi_target": phi_target,
            "phi_target_raw": phi_target_raw,
            "phi_clipped_by_max": phi_clipped_by_max,
            "theta": theta,
            "valid": pred_valid,
            "used_tangent": used_tangent,
        }

    def __pure_curvature_predictor(self):
        u_tan_bar, lmbda_tan_bar, tangent_valid = (
            self.__compute_tangent_direction()
        )
        if not tangent_valid:
            raise RuntimeError("Approximate tangent direction is invalid.")

        direction = self.__pure_curvature_direction(
            u_tan_bar, lmbda_tan_bar
        )

        if not direction["valid"]:
            direction["u_pred_bar"] = u_tan_bar
            direction["lmbda_pred_bar"] = lmbda_tan_bar
            mode = "tangent fallback"
        elif direction["used_tangent"]:
            mode = "tangent limit"
        elif direction["phi_clipped_by_max"]:
            mode = "phi_max clipped"
        else:
            mode = "SATCP"

        u_pred = self.u_n + self.delta_s * direction["u_pred_bar"]
        lmbda_pred = (
            self.lmbda_n + self.delta_s * direction["lmbda_pred_bar"]
        )

        info = {
            "mode": mode,
            "theta": direction["theta"],
            "kappa_now": direction["kappa_now"],
            "kappa_rate": direction["kappa_rate"],
            "phi_target_raw": direction["phi_target_raw"],
            "phi_target": direction["phi_target"],
            "phi_max": self.phi_max,
            "phi_clipped_by_max": direction["phi_clipped_by_max"],
        }
        return u_pred, lmbda_pred, direction["kappa_now"], True, info

    def __update_latest_increment(self, u_new, lmbda_new):
        self.delta_u_n = u_new - self.u_n
        self.delta_lmbda_n = lmbda_new - self.lmbda_n
        (
            self.delta_u_n_bar,
            self.delta_lmbda_n_bar,
            valid,
            _,
        ) = self.__standardize_increment(
            self.delta_u_n, self.delta_lmbda_n
        )
        if not valid:
            raise RuntimeError(
                "The newly accepted increment has zero or invalid arc norm."
            )
        self.has_delta_u_n = True

    # ------------------------------------------------------------------
    # Predictor selection for one attempted increment
    # ------------------------------------------------------------------
    def __set_trial_predictor(self):
        accepted_step_index = self.counter + 1
        is_first_accepted_step = self.counter == 0

        if is_first_accepted_step:
            # First-step predictor: retain the previous accepted state and
            # increment the load factor by Ds.
            self.__update_nodal_values(self.u_n)
            self.lmbda = self.lmbda_n + self.delta_s
            self.last_predictor_info = {
                "mode": "first-step load control",
                "delta_lmbda": self.delta_s,
            }
            return accepted_step_index, self.kappa_prev, False

        use_startup_sep = (
            self.counter
            < self.num_accepted_increments_before_curvature
            or not self.has_delta_u_n
            or not self.has_du_1_last
        )

        if use_startup_sep:
            alpha = self.delta_s / self.delta_s_n
            self.__update_nodal_values(
                (1.0 + alpha) * self.u_n - alpha * self.u_n_1
            )
            self.lmbda = (
                (1.0 + alpha) * self.lmbda_n
                - alpha * self.lmbda_n_1
            )
            self.last_predictor_info = {
                "mode": "start-up SEP",
                "alpha": alpha,
            }
            return accepted_step_index, self.kappa_prev, False

        (
            u_pred,
            lmbda_pred,
            kappa_trial,
            has_kappa_trial,
            predictor_info,
        ) = self.__pure_curvature_predictor()
        self.__update_nodal_values(u_pred)
        self.lmbda = lmbda_pred
        self.last_predictor_info = predictor_info
        return accepted_step_index, kappa_trial, has_kappa_trial

    # ------------------------------------------------------------------
    # Public one-attempt solve operation
    # ------------------------------------------------------------------
    def solve(self):
        """
        Perform exactly one attempted increment.

        Returns
        -------
        bool
            True if the attempted increment converged and was accepted;
            False if it failed and the arc length was reduced for restart.
        """
        self.last_iters = 0
        self.last_corrections = 0
        self.attempted_steps += 1

        print("\nAttempted arc-length step", self.attempted_steps)

        accepted_step_index, kappa_trial, has_kappa_trial = (
            self.__set_trial_predictor()
        )
        self.__set_load_factor(self.lmbda)

        delta_u = self.u.vector().copy() - self.u_n
        delta_lmbda = self.lmbda - self.lmbda_n

        self.converged_prev = self.converged
        self.converged = False

        # Trial unit-load response.  It is overwritten only by a response
        # that is actually used in a Newton correction and committed only if
        # this attempted step is accepted.
        du_1_step = None

        for solver_iter in range(1, self.max_iter + 1):
            K, R = self.__assemble_system_current()

            if accepted_step_index > 1:
                a = 2.0 * delta_u
                b = 2.0 * self.psi * delta_lmbda * self.FF
                A = (
                    delta_u.inner(delta_u)
                    + self.psi * delta_lmbda ** 2 * self.FF
                    - self.delta_s ** 2
                )
            else:
                # First accepted increment: standard Newton with prescribed
                # load increment. This is the prescribed-load split-solver branch.
                a = self.__zero_like(delta_u)
                b = 1.0
                A = 0.0

            R_norm = R.norm("l2")
            combined_norm = sqrt(R_norm ** 2 + A ** 2)

            # Report the number of corrections already applied, matching the
            # existing point-load solvers' 0-based Newton iteration log.
            logged_iter = solver_iter - 1

            print(
                "Iteration: {0:d}\n"
                "|Total Norm: {1:.4e} |Residual: {2:.4e} |A: {3:.4e} "
                "|delta_u_norm: {4:.4e} |delta_lmbda: {5:.4e}".format(
                    logged_iter,
                    combined_norm,
                    R_norm,
                    A,
                    delta_u.norm("l2"),
                    delta_lmbda,
                )
            )

            # IMPORTANT: convergence is checked before solving for du_1.
            # Therefore du_1_step remains the response used in the last
            # actual correction, consistent with the SATCP history definition.
            if combined_norm < self.abs_tol:
                self.converged = True
                print("Load factor = {0:.6e}".format(self.lmbda))
                break

            du_1 = Vector()
            du_2 = Vector()
            solve(K, du_1, self.F_ext_vec, self.solver)
            solve(K, du_2, R, self.solver)

            denominator = b + a.inner(du_1)
            if not isfinite(denominator) or abs(denominator) <= self.denominator_tol:
                print(
                    "Split-equation denominator is zero/invalid; "
                    "the attempted step is rejected."
                )
                break

            dlmbda = (a.inner(du_2) - A) / denominator
            du = -du_2 + dlmbda * du_1

            # Save only a unit-load response that is actually used in this
            # correction.  A later convergence check never overwrites it.
            du_1_step = du_1.copy()

            delta_u += du
            delta_lmbda += dlmbda
            self.__update_nodal_values(self.u.vector() + du)
            self.lmbda += dlmbda
            self.__set_load_factor(self.lmbda)

            # Count one actual Newton correction.  The final convergence
            # check on the next loop pass therefore adds no extra iteration.
            self.last_iters += 1
            self.total_iters += 1
            self.last_corrections += 1
            self.total_corrections += 1

        if self.converged:
            # The first accepted step establishes the actual arc-length scale.
            if accepted_step_index == 1:
                self.delta_s = sqrt(
                    delta_u.inner(delta_u)
                    + self.psi * delta_lmbda ** 2 * self.FF
                )
                if (
                    not isfinite(self.delta_s)
                    or self.delta_s <= self.norm_tol
                ):
                    raise RuntimeError(
                        "The first accepted increment has zero/invalid arc length."
                    )
                self.delta_s_max = self.delta_s
                self.delta_s_min = self.delta_s / 1024.0

            ds_accepted = self.delta_s
            self.__record_accepted_arc_length(ds_accepted)

            # Trial--commit semantics: failed attempts never modify these
            # accepted predictor histories.
            if du_1_step is not None:
                self.du_1_last = du_1_step.copy()
                self.has_du_1_last = True

            if has_kappa_trial:
                self.kappa_prev = kappa_trial
                self.has_kappa_prev = True

            u_new = self.u.vector().copy()
            lmbda_new = self.lmbda
            self.__update_latest_increment(u_new, lmbda_new)

            self.u_n_1 = self.u_n.copy()
            self.lmbda_n_1 = self.lmbda_n
            self.u_n = u_new
            self.lmbda_n = lmbda_new

            self.delta_s_n = ds_accepted
            self.counter += 1

            # Increase the arc length only when the immediately preceding
            # attempted increment also converged.
            if self.converged_prev:
                self.delta_s = min(
                    max(2.0 * self.delta_s, self.delta_s_min),
                    self.delta_s_max,
                )

        else:
            self.restart_count += 1
            if self.converged_prev:
                self.delta_s = max(
                    0.5 * self.delta_s, self.delta_s_min
                )
            else:
                self.delta_s = max(
                    0.25 * self.delta_s, self.delta_s_min
                )
            print(
                "Restart: new arc length Ds = {0:.6e}".format(self.delta_s)
            )

        return self.converged
