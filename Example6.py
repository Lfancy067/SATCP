# SPDX-FileCopyrightText: 2026 Fan Liu (modifications)
# SPDX-License-Identifier: LGPL-3.0-or-later
#
# Adapted from the FEniCS-Shells demonstration:
#   demo_nonlinear-naghdi-cylindrical.py
#   https://fenics-shells.readthedocs.io/en/latest/demo/
#   nonlinear-naghdi-cylindrical/demo_nonlinear-naghdi-cylindrical.py.html
#
# Upstream project:
#   FEniCS-Shells
#   https://bitbucket.org/unilucompmech/fenics-shells
#
# Modified by Fan Liu in 2026.
# Modifications:
# - pinched semi-cylindrical-shell benchmark;
# - half-model geometry and boundary conditions;
# - concentrated point-load treatment;
# - coupling with the SATCP arc-length solver;
# - accepted-step output and continuation controls.
#
# Relevant reference:
# J. S. Hale, M. Brunetti, S. P. A. Bordas, and C. Maurini,
# "Simple and extensible plate and shell finite element models through
# automatic code generation tools", Computers & Structures,
# 209 (2018), 163-181.
#
# This modified file is distributed under GNU LGPL-3.0-or-later.
# See LICENSES/LGPL-3.0-or-later.txt and
# LICENSES/GPL-3.0-or-later.txt.

import os
import sys
import time
from math import pi
from pathlib import Path

from dolfin import *
from ufl import Index


parameters["form_compiler"]["quadrature_degree"] = 4


# -----------------------------------------------------------------------------
# Run options and output paths
# -----------------------------------------------------------------------------
SAVE_DAT = False
SAVE_DISPLACEMENT = False

DATA_DIR = "data"
DATA_FILENAME = "Example6_present.dat"
DISPLACEMENT_DIR = "displacement"
DISPLACEMENT_FILENAME = "Example6_displacement.pvd"

P_REF = 500.0               # Reference point load applied to the half model
LOAD_DIRECTION_SIGN = -1.0  # Negative global z direction

NSTEP_MAX = 10000
MAX_FAILED_ATTEMPTS = 100
END_ACCEPTED_ARC_LENGTH_BY_DS0 = 40.0
ACCEPTED_ARC_LENGTH_STOP_TOL = 1.0e-10


t0 = time.process_time()


# -----------------------------------------------------------------------------
# Geometry, material, and mesh
# -----------------------------------------------------------------------------
# This script analyses one half of a pinched semi-cylindrical shell. The shell
# has radius rho, axial length L, uniform thickness t, Young's modulus E, and
# Poisson's ratio nu. One circumferential end is fully clamped, whereas the
# opposite end is free and carries a concentrated transverse load at its
# midpoint. Symmetry of the geometry and loading about the longitudinal
# mid-plane permits use of the half model.

rho = 1.016
L = 3.048
E, nu = 2.0685e7, 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = 2.0 * mu * nu / (1.0 - 2.0 * nu)
t = Constant(0.03)
LOAD_POINT = (0.0, L)  # Midpoint A of the free circumferential edge

# The parametric domain is
#
#     omega = [0, pi/2] x [0, L],
#
# where x_0 is the circumferential angle and x_1 is the axial coordinate.
# A 16 x 32 rectangular mesh is used, with each rectangle divided into two
# triangles, giving 1024 triangular shell elements.

nx = 16
ny = 32
mesh = RectangleMesh(Point(0.0, 0.0), Point(pi / 2.0, L), nx, ny)

# Analytical reference midsurface of the half-cylinder.
initial_shape = Expression(
    ("r*sin(x[0])", "x[1]", "r*cos(x[0])"),
    r=rho,
    degree=4,
)
V_phi = FunctionSpace(mesh, VectorElement("P", triangle, degree=2, dim=3))
phi0 = project(initial_shape, V_phi)


# Compute the unit normal to the reference midsurface and project it onto a
# linear vector finite element space.
def normal(y):
    n = cross(y.dx(0), y.dx(1))
    return n / sqrt(inner(n, n))


V_normal = FunctionSpace(mesh, VectorElement("P", triangle, degree=1, dim=3))
n0 = project(normal(phi0), V_normal)


# The five-parameter Naghdi shell kinematics is described by the midsurface
# displacement field u and the unit director d. The current midsurface is
# phi = phi_0 + u. The director orientation is parametrised by two angles
# beta, which enforce the unit-length condition explicitly; see Betsch
# et al. (1998).
def director(beta):
    return as_vector(
        [
            sin(beta[1]) * cos(beta[0]),
            -sin(beta[0]),
            cos(beta[1]) * cos(beta[0]),
        ]
    )


# In the reference configuration, the director is assumed to coincide with
# the midsurface unit normal. The corresponding initial director angles
# beta_0 are therefore obtained from n_0.
beta0_expression = Expression(
    [
        "atan2(-n[1], sqrt(pow(n[0],2) + pow(n[2],2)))",
        "atan2(n[0], n[2])",
    ],
    n=n0,
    degree=4,
)
V_beta = FunctionSpace(mesh, VectorElement("P", triangle, degree=2, dim=2))
beta0 = project(beta0_expression, V_beta)
d0 = director(beta0)


# -----------------------------------------------------------------------------
# Mixed function space and nonlinear Naghdi-shell energy
# -----------------------------------------------------------------------------
# Following the FEniCS-Shells nonlinear Naghdi-shell formulation described
# by Hale et al. (2018), the displacement field is approximated using
# [P_2 + B_3]^3 and the director-angle field using [P_2]^2. The two fields
# are collected in the mixed state vector q = (u, beta).

P2 = FiniteElement("Lagrange", triangle, degree=2)
bubble = FiniteElement("B", triangle, degree=3)
enriched = P2 + bubble
element = MixedElement(
    [VectorElement(enriched, dim=3), VectorElement(P2, dim=2)]
)
Q = FunctionSpace(mesh, element)

q_ = Function(Q)
q = TrialFunction(Q)
q_t = TestFunction(Q)
u_, beta_ = split(q_)

# Kinematic quantities in the current configuration.
F = grad(u_) + grad(phi0)
d = director(beta_ + beta0)

# Reference metric tensor a_0 and curvature tensor b_0.
a0 = grad(phi0).T * grad(phi0)
b0 = -0.5 * (grad(phi0).T * grad(d0) + grad(d0).T * grad(phi0))

# Membrane, bending, and transverse-shear strain measures.
def membrane_strain(F_):
    return 0.5 * (F_.T * F_ - a0)


def bending_strain(F_, d_):
    return -0.5 * (F_.T * grad(d_) + grad(d_).T * F_) - b0


def shear_strain(F_, d_):
    return F_.T * d_ - grad(phi0).T * d0


# Using the contravariant reference metric a0_contra, represent the isotropic
# plane-stress constitutive relation by the fourth-order contravariant tensor
# A. UFL index notation is used for tensor operations.
a0_contra = inv(a0)
j0 = det(a0)

i, j, l, m = Index(), Index(), Index(), Index()
A_ = as_tensor(
    (
        ((2.0 * lmbda * mu) / (lmbda + 2.0 * mu))
        * a0_contra[i, j]
        * a0_contra[l, m]
        + mu
        * (
            a0_contra[i, l] * a0_contra[j, m]
            + a0_contra[i, m] * a0_contra[j, l]
        )
    ),
    [i, j, l, m],
)

# Membrane-force resultant N, bending-moment resultant M, and transverse-
# shear resultant T, expressed as Lagrangian shell stress resultants.
N = as_tensor(t * A_[i, j, l, m] * membrane_strain(F)[l, m], [i, j])
M = as_tensor(
    (t**3 / 12.0) * A_[i, j, l, m] * bending_strain(F, d)[l, m],
    [i, j],
)
T = as_tensor(t * mu * a0_contra[i, j] * shear_strain(F, d)[j], [i])

# Membrane, bending, and transverse-shear energy densities per unit area of
# the reference midsurface.
psi_m = 0.5 * inner(N, membrane_strain(F))
psi_b = 0.5 * inner(M, bending_strain(F, d))
psi_s = 0.5 * inner(T, shear_strain(F, d))

# Partial selective reduced integration (PSRI) is used to alleviate membrane
# and transverse-shear locking, following the strategy adopted in the
# FEniCS-Shells nonlinear Naghdi-shell demonstration and the locking-treatment
# principles of Arnold and Brezzi (1997).
#
# The membrane and shear energies are split into two weighted parts. The part
# weighted by 1-alpha is integrated using quadrature degree 2, whereas the
# remaining terms and the bending energy use quadrature degree 4. The
# weighting parameter is alpha = (t/h)^2, where h is the local cell diameter.
dx_h = dx(metadata={"quadrature_degree": 2})
h = CellDiameter(mesh)
alpha = project(t**2 / h**2, FunctionSpace(mesh, "DG", 0))

Pi = (
    psi_b * sqrt(j0) * dx
    + alpha * psi_m * sqrt(j0) * dx
    + alpha * psi_s * sqrt(j0) * dx
    + (1.0 - alpha) * psi_s * sqrt(j0) * dx_h
    + (1.0 - alpha) * psi_m * sqrt(j0) * dx_h
)

dPi = derivative(Pi, q_, q_t)
J = derivative(dPi, q_, q)


# -----------------------------------------------------------------------------
# Boundary conditions
# -----------------------------------------------------------------------------
# Boundary conditions for the half-shell model:
#
# - x_1 = 0: fully clamped circumferential edge;
# - x_1 = L: free circumferential edge;
# - x_0 = 0: longitudinal symmetry plane, with u_x = 0 and beta_1 = 0;
# - x_0 = pi/2: physical longitudinal edge, with u_z = 0 and beta_1 = 0.
#
# The last two conditions reproduce the symmetry and longitudinal-edge
# restraints of the full semi-cylindrical shell.

clamped_boundary = lambda x, on_boundary: near(x[1], 0.0, 1.0e-6) and on_boundary
symmetry_boundary = lambda x, on_boundary: near(x[0], 0.0, 1.0e-6) and on_boundary
longitudinal_edge_boundary = (
    lambda x, on_boundary: near(x[0], pi / 2.0, 1.0e-6) and on_boundary
)

zero_state = project(Constant((0.0, 0.0, 0.0, 0.0, 0.0)), Q)
zero_scalar_beta = project(Constant(0.0), Q.sub(1).sub(1).collapse())
zero_scalar_ux = project(Constant(0.0), Q.sub(0).sub(0).collapse())
zero_scalar_uz = project(Constant(0.0), Q.sub(0).sub(2).collapse())

bc_clamped = DirichletBC(Q, zero_state, clamped_boundary)
bc_symmetry_u = DirichletBC(
    Q.sub(0).sub(0), zero_scalar_ux, symmetry_boundary
)
bc_symmetry_beta = DirichletBC(
    Q.sub(1).sub(1), zero_scalar_beta, symmetry_boundary
)
bc_longitudinal_u = DirichletBC(
    Q.sub(0).sub(2), zero_scalar_uz, longitudinal_edge_boundary
)
bc_longitudinal_beta = DirichletBC(
    Q.sub(1).sub(1), zero_scalar_beta, longitudinal_edge_boundary
)

bcs = [
    bc_clamped,
    bc_symmetry_u,
    bc_symmetry_beta,
    bc_longitudinal_u,
    bc_longitudinal_beta,
]


# -----------------------------------------------------------------------------
# Reference point load and SATCP arc-length solver
# -----------------------------------------------------------------------------
# Assemble the reference point-load vector for the half model. The load is
# applied at point A, the midpoint of the free circumferential edge. The full-
# shell load reported below is P = 2*lambda*P_REF.
def build_reference_point_load(function_space, state, boundary_conditions):
    f_ext = state.vector().copy()
    f_ext.zero()
    PointSource(
        function_space.sub(0).sub(2),
        Point(LOAD_POINT[0], LOAD_POINT[1]),
        LOAD_DIRECTION_SIGN * P_REF,
    ).apply(f_ext)
    for bc in boundary_conditions:
        bc.apply(f_ext)
    return f_ext


F_ext_vec = build_reference_point_load(Q, q_, bcs)

from Present_solver import force_control

# Pass the complete mixed state q_ (midsurface displacements and director
# angles) to the arc-length solver, rather than the displacement subfield u_.
arc_solver = force_control(
    psi=1.0,
    lmbda0=0.05,
    max_iter=15,
    u=q_,
    F_int=dPi,
    F_ext=F_ext_vec,
    bcs=bcs,
    J=J,
    load_factor=None,
    abs_tol=1.0e-6,
    rel_tol=1.0e-8,
    solver="mumps",
)


# -----------------------------------------------------------------------------
# Output setup and initial state
# -----------------------------------------------------------------------------
dat_path = os.path.join(DATA_DIR, DATA_FILENAME)
displacement_path = os.path.join(DISPLACEMENT_DIR, DISPLACEMENT_FILENAME)

file_u = None
if SAVE_DISPLACEMENT:
    os.makedirs(DISPLACEMENT_DIR, exist_ok=True)
    file_u = File(displacement_path)

if SAVE_DAT:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(dat_path, "w") as data_file:
        data_file.write("# load_N\tdownward_displacement_at_A\n")

q_.assign(zero_state)

attempt = 0
converged_step = 0
current_step_cutbacks = 0
total_cutbacks = 0
total_iters = 0


def solve_and_count_iterations(solver):
    """Run one attempted increment and return its actual correction count."""
    ok = solver.solve()
    return ok, solver.last_iters


# -----------------------------------------------------------------------------
# Arc-length continuation loop
# -----------------------------------------------------------------------------
# NSTEP_MAX is the maximum number of attempted increments. Only accepted
# states are written, and rejected attempts are restored to the last accepted
# equilibrium state.
while attempt < NSTEP_MAX:
    attempt += 1

    # Save the current accepted state before each attempted increment.
    q_backup = q_.vector().copy()
    lambda_backup = arc_solver.lmbda

    ok, niter = solve_and_count_iterations(arc_solver)
    total_iters += niter

    if not ok:
        current_step_cutbacks += 1
        total_cutbacks += 1
        print(
            "attempt %d/%d: not converged, "
            "converged_step=%d, cutbacks_this_step=%d, cutbacks_total=%d"
            % (
                attempt,
                NSTEP_MAX,
                converged_step,
                current_step_cutbacks,
                total_cutbacks,
            )
        )

        # Restore the last accepted state after a rejected attempt.
        q_.vector().set_local(q_backup.get_local())
        q_.vector().apply("insert")
        arc_solver.lmbda = lambda_backup

        if current_step_cutbacks >= MAX_FAILED_ATTEMPTS:
            print("Too many failed attempts on this step. Stop.")
            break

        # Do not write output for a rejected attempt; retain it in the
        # attempted-step count.
        continue

    converged_step += 1

    u_vis = project(u_, V_phi)
    w_z = u_vis(LOAD_POINT[0], LOAD_POINT[1])[2]
    downward_displacement = LOAD_DIRECTION_SIGN * w_z
    full_shell_load = 2.0 * arc_solver.lmbda * P_REF
    accepted_arc_length_by_ds0 = arc_solver.get_accepted_arc_length(
        normalized=True
    )

    if SAVE_DISPLACEMENT:
        u_vis.rename("u", "u")
        file_u << (u_vis, arc_solver.lmbda)

    # Append accepted states only to the data file.
    if SAVE_DAT:
        with open(dat_path, "a") as data_file:
            data_file.write(
                "%.12e\t%.12e\n"
                % (full_shell_load, downward_displacement)
            )

    print(
        "attempt=%d, converged_step=%d, lambda=%.6e, "
        "load=%.6e, downward_displacement=%.6e, iters=%d, "
        "accepted_arc_length=%.12f*Ds0, cutbacks_this_step=%d, "
        "cutbacks_total=%d"
        % (
            attempt,
            converged_step,
            arc_solver.lmbda,
            full_shell_load,
            downward_displacement,
            niter,
            accepted_arc_length_by_ds0,
            current_step_cutbacks,
            total_cutbacks,
        )
    )

    if (
        accepted_arc_length_by_ds0 + ACCEPTED_ARC_LENGTH_STOP_TOL
        >= END_ACCEPTED_ARC_LENGTH_BY_DS0
    ):
        print(
            "Stop: accepted arc length reached %.12f*Ds0 "
            "(limit=%.12f*Ds0)."
            % (
                accepted_arc_length_by_ds0,
                END_ACCEPTED_ARC_LENGTH_BY_DS0,
            )
        )
        break

    # Reset the current-step cutback counter after acceptance.
    current_step_cutbacks = 0


t1 = time.process_time()
print("\nTotal iters = %d" % total_iters)
print("Total attempts = %d" % attempt)
print("Total converged steps = %d" % converged_step)
print("Total cutbacks = %d" % total_cutbacks)
print(
    "Accepted arc length = %.12f*Ds0"
    % arc_solver.get_accepted_arc_length(normalized=True)
)
print("CPU time = %.6f s" % (t1 - t0))

if SAVE_DAT:
    print("dat file written to: %s" % dat_path)
else:
    print("dat file saving disabled")

if SAVE_DISPLACEMENT:
    print("displacement pvd written to: %s" % displacement_path)
else:
    print("displacement field saving disabled")
