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
# - hinged cylindrical-panel benchmark;
# - quarter-model geometry and boundary conditions;
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
from pathlib import Path
import matplotlib.pyplot as plt
from dolfin import *
from ufl import Index
parameters["form_compiler"]["quadrature_degree"] = 4


# -----------------------------------------------------------------------------
# Run options and output paths
# -----------------------------------------------------------------------------
SAVE_DAT = False
SAVE_DISPLACEMENT = False
SAVE_FIGURES = False


DATA_DIR = "data"
DATA_FILENAME = "Example7_present_t12_7.dat"
DISPLACEMENT_DIR = "displacement"
DISPLACEMENT_FILENAME = "displacement.pvd"
FIGURE_DIR = "figures"
MESH_FIGURE_FILENAME = "mesh.png"
INITIAL_FIGURE_FILENAME = "initial_configuration.png"

P_REF = 250.0              # Reference load on the quarter model
LOAD_POINT = (0.0, 0.0)    # Centre point of the full panel
LOAD_DIRECTION_SIGN = -1.0 # Negative global z direction

t0 = time.process_time()

# This script analyses one quarter of a hinged cylindrical panel with
# radius rho, full longitudinal length 2*L, half-opening angle theta,
# and uniform thickness t. The panel is made of a homogeneous, isotropic,
# linearly elastic material with Young's modulus E and Poisson's ratio nu.
# A plane-stress constitutive relation is adopted for the shell model.

rho = 2540          # Panel radius
L = 254             # Quarter-model longitudinal length; full length = 2*L
E, nu = 3102.75, 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = 2.0 * mu * nu / (1.0 - 2.0 * nu)
t = Constant(12.7)  # Shell thickness
theta = 0.1         # Half-opening angle in radians
ns = 16
nL = 16
nstep = 10000
END_ACCEPTED_ARC_LENGTH_BY_DS0 = 60.0
ACCEPTED_ARC_LENGTH_STOP_TOL = 1.0e-10
# The stress-free midsurface phi_0 is represented analytically in terms of
# the circumferential arc-length coordinate x_0 and longitudinal coordinate
# x_1. For the quarter model, the parametric domain is
#
#     omega = [0, rho*theta] x [0, L].
#
# The boundaries x_0 = 0 and x_1 = 0 are symmetry planes, x_0 = rho*theta
# is the hinged longitudinal edge, and x_1 = L is a free curved edge.

P1, P2 = Point(0.0, 0.0), Point(rho*theta, L)
mesh = RectangleMesh(P1, P2, ns, nL)
if SAVE_FIGURES:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plot(mesh)
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$x_1$")
    plt.savefig(os.path.join(FIGURE_DIR, MESH_FIGURE_FILENAME))
    plt.close()

# Represent the analytical reference midsurface on a quadratic vector
# finite element space.


initial_shape = Expression(('r*sin(x[0]/r)', 'x[1]', 'r*cos(x[0]/r)'),r=rho, degree=4)
V_phi = FunctionSpace(mesh, VectorElement("P", triangle, degree=2, dim=3))
phi0 = project(initial_shape, V_phi)


# Compute the unit normal to the reference midsurface and project it onto
# a linear vector finite element space.

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
    return as_vector([sin(beta[1]) * cos(beta[0]), -sin(beta[0]), cos(beta[1]) * cos(beta[0])])


# In the reference configuration, the director is assumed to coincide with
# the midsurface unit normal. The corresponding initial director angles
# beta_0 are therefore obtained from n_0.

beta0_expression = Expression(["atan2(-n[1], sqrt(pow(n[0],2) + pow(n[2],2)))",
                               "atan2(n[0],n[2])"], n=n0, degree=4)

V_beta = FunctionSpace(mesh, VectorElement("P", triangle, degree=2, dim=2))
beta0 = project(beta0_expression, V_beta)

# Construct the reference director field.

d0 = director(beta0)


# Optional utility for visualising the shell shape and director field.

def plot_shell(y, n=None):
    y_0, y_1, y_2 = y.split(deepcopy=True)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(y_0.compute_vertex_values(),
                    y_1.compute_vertex_values(),
                    y_2.compute_vertex_values(),
                    triangles=y.function_space().mesh().cells(),
                    linewidth=1, antialiased=True, shade=False)
    if n:
        n_0, n_1, n_2 = n.split(deepcopy=True)
        ax.quiver(y_0.compute_vertex_values(),
                  y_1.compute_vertex_values(),
                  y_2.compute_vertex_values(),
                  n_0.compute_vertex_values(),
                  n_1.compute_vertex_values(),
                  n_2.compute_vertex_values(),
                  length=.2, color="r")
    ax.view_init(elev=20, azim=80)
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$x_1$")
    plt.xticks([-1, 0, 1])
    plt.yticks([0, pi / 2])
    return ax


if SAVE_FIGURES:
    plot_shell(phi0, project(d0, V_normal))
    plt.savefig(os.path.join(FIGURE_DIR, INITIAL_FIGURE_FILENAME))
    plt.close()

# Following the FEniCS-Shells nonlinear Naghdi-shell formulation described
# by Hale et al. (2018), the displacement field is approximated using
# [P_2 + B_3]^3 and the director-angle field using [P_2]^2. The two fields
# are collected in the mixed state vector q = (u, beta).

P2 = FiniteElement("Lagrange", triangle, degree=2)
bubble = FiniteElement("B", triangle, degree=3)
enriched = P2 + bubble

element = MixedElement([VectorElement(enriched, dim=3), VectorElement(P2, dim=2)])

Q = FunctionSpace(mesh, element)

# Define the mixed state, trial field, and test field, then split the state
# into its displacement and director-angle components.

q_, q, q_t = Function(Q), TrialFunction(Q), TestFunction(Q)
u_, beta_ = split(q_)

# Kinematic quantities in the current configuration.

F = grad(u_) + grad(phi0)
d = director(beta_ + beta0)

# Reference metric tensor a_0 and curvature tensor b_0.

a0 = grad(phi0).T * grad(phi0)
b0 = -0.5 * (grad(phi0).T * grad(d0) + grad(d0).T * grad(phi0))

# Membrane, bending, and transverse-shear strain measures.

e = lambda F: 0.5 * (F.T * F - a0)
k = lambda F, d: -0.5 * (F.T * grad(d) + grad(d).T * F) - b0
gamma = lambda F, d: F.T * d - grad(phi0).T * d0

# Using the contravariant reference metric a0_contra, represent the
# isotropic plane-stress constitutive relation by the fourth-order
# contravariant tensor A. UFL index notation is used for tensor operations.

a0_contra = inv(a0)
j0 = det(a0)

i, j, l, m = Index(), Index(), Index(), Index()
A_ = as_tensor((((2.0 * lmbda * mu) / (lmbda + 2.0 * mu)) * a0_contra[i, j] * a0_contra[l, m]
                + 1.0 * mu * (a0_contra[i, l] * a0_contra[j, m] + a0_contra[i, m] * a0_contra[j, l]))
               , [i, j, l, m])

# Membrane-force resultant N, bending-moment resultant M, and transverse-
# shear resultant T, expressed as Lagrangian shell stress resultants.

N = as_tensor(t * A_[i, j, l, m] * e(F)[l, m], [i, j])
M = as_tensor((t ** 3 / 12.0) * A_[i, j, l, m] * k(F, d)[l, m], [i, j])
T = as_tensor(t * mu * a0_contra[i, j] * gamma(F, d)[j], [i])

# Membrane, bending, and transverse-shear energy densities per unit area
# of the reference midsurface.

psi_m = 0.5 * inner(N, e(F))
psi_b = 0.5 * inner(M, k(F, d))
psi_s = 0.5 * inner(T, gamma(F, d))

# Partial selective reduced integration (PSRI) is used to alleviate
# membrane and transverse-shear locking, following the strategy adopted in
# the FEniCS-Shells nonlinear Naghdi-shell demonstration and the locking-
# treatment principles of Arnold and Brezzi (1997).
#
# The membrane and shear energies are split into two weighted parts. The
# part weighted by 1-alpha is integrated using quadrature degree 2, whereas
# the remaining terms and the bending energy use quadrature degree 4. The
# weighting parameter is alpha = (t/h)^2, where h is the local cell diameter.

dx_h = dx(metadata={'quadrature_degree': 2})
h = CellDiameter(mesh)
alpha = project(t ** 2 / h ** 2, FunctionSpace(mesh, 'DG', 0))

Pi_PSRI = psi_b * sqrt(j0) * dx + alpha * psi_m * sqrt(j0) * dx + alpha * psi_s * sqrt(j0) * dx + (
            1.0 - alpha) * psi_s * sqrt(j0) * dx_h + (1.0 - alpha) * psi_m * sqrt(j0) * dx_h

# Total elastic energy, first variation, and consistent tangent.

Pi = Pi_PSRI
dPi = derivative(Pi, q_, q_t)
J = derivative(dPi, q_, q)

# Boundary conditions for the quarter-panel model:
#
# - x_0 = rho*theta: hinged longitudinal edge; all three displacement
#   components are constrained, while the director rotations remain free;
# - x_0 = 0: circumferential symmetry boundary, with u_0 = 0 and beta_1 = 0;
# - x_1 = 0: longitudinal symmetry boundary, with u_1 = 0 and beta_0 = 0;
# - x_1 = L: free curved edge, represented by natural boundary conditions.

symmetry_x_boundary = lambda x, on_boundary: near(x[0], 0.0, 1.e-6) and on_boundary
symmetry_y_boundary = lambda x, on_boundary: near(x[1], 0.0, 1.e-6) and on_boundary
supported_boundary = lambda x, on_boundary: near(x[0], rho*theta, 1.e-6) and on_boundary

bc_u_supported = DirichletBC(
    Q.sub(0),
    project(Constant((0.0, 0.0, 0.0)), Q.sub(0).collapse()),
    supported_boundary
)

bc_symmetry_x_u = DirichletBC(
    Q.sub(0).sub(0),
    project(Constant(0.0), Q.sub(0).sub(0).collapse()),
    symmetry_x_boundary
)
bc_symmetry_x_beta = DirichletBC(
    Q.sub(1).sub(1),
    project(Constant(0.0), Q.sub(1).sub(1).collapse()),
    symmetry_x_boundary
)
bc_symmetry_y_u = DirichletBC(
    Q.sub(0).sub(1),
    project(Constant(0.0), Q.sub(0).sub(1).collapse()),
    symmetry_y_boundary
)
bc_symmetry_y_beta = DirichletBC(
    Q.sub(1).sub(0),
    project(Constant(0.0), Q.sub(1).sub(0).collapse()),
    symmetry_y_boundary
)

bcs = [
    bc_u_supported,
    bc_symmetry_x_u,
    bc_symmetry_x_beta,
    bc_symmetry_y_u,
    bc_symmetry_y_beta,
]

print("supported bc dofs:", len(bc_u_supported.get_boundary_values()))
print("symmetry x u dofs:", len(bc_symmetry_x_u.get_boundary_values()))
print("symmetry x beta dofs:", len(bc_symmetry_x_beta.get_boundary_values()))
print("symmetry y u dofs:", len(bc_symmetry_y_u.get_boundary_values()))
print("symmetry y beta dofs:", len(bc_symmetry_y_beta.get_boundary_values()))


# Assemble the reference point-load vector for the quarter model. The load
# is applied at the intersection of the two symmetry boundaries. The full-
# panel load reported below is P = 4*lambda*P_REF.
def build_reference_point_load(Q, q_, bcs):
    f_ext = q_.vector().copy()
    f_ext.zero()
    PointSource(
        Q.sub(0).sub(2),
        Point(LOAD_POINT[0], LOAD_POINT[1]),
        LOAD_DIRECTION_SIGN * P_REF,
    ).apply(f_ext)
    for bc in bcs:
        bc.apply(f_ext)
    return f_ext


F_ext_vec = build_reference_point_load(Q, q_, bcs)

from Present_solver import force_control
# Pass the complete mixed state q_ (midsurface displacements and director
# angles) to the arc-length solver, rather than the displacement subfield u_.
arc_solver = force_control(
    psi=1.0,
    lmbda0=0.1,
    max_iter=10,
    u=q_,
    F_int=dPi,
    F_ext=F_ext_vec,
    bcs=bcs,
    J=J,
    load_factor=None,
    abs_tol=1e-6,
    rel_tol=1e-8,
    solver="mumps"
)

# -------------------------------------------------
dat_path = os.path.join(DATA_DIR, DATA_FILENAME)
displacement_path = os.path.join(DISPLACEMENT_DIR, DISPLACEMENT_FILENAME)

file_u = None
if SAVE_DISPLACEMENT:
    os.makedirs(DISPLACEMENT_DIR, exist_ok=True)
    file_u = File(displacement_path)

if SAVE_DAT:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(dat_path, "w") as f:
        f.write("# load\tdisplacement_in_load_direction\n")

# -------------------------------------------------
# Initial state and continuation counters
# -------------------------------------------------
q_.assign(project(Constant((0.0, 0.0, 0.0, 0.0, 0.0)), Q))


attempt = 0
converged_step = 0

current_step_cutbacks = 0
total_cutbacks = 0
total_iters = 0
max_failed_attempts = 100


def solve_and_count_iterations(arc_solver):
    """Run one attempted increment and return the actual correction count."""
    ok = arc_solver.solve()
    return ok, arc_solver.last_iters

# -------------------------------------------------
# Arc-length continuation loop. nstep is the maximum number of attempted
# increments. Only accepted states are written, and rejected attempts are
# restored to the last accepted equilibrium state.
# -------------------------------------------------

while attempt < nstep:
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
            f"attempt {attempt}: not converged, "
            f"converged_step={converged_step}, "
            f"cutbacks_this_step={current_step_cutbacks}, "
            f"cutbacks_total={total_cutbacks}"
        )

        # Restore the last accepted state after a rejected attempt.
        q_.vector().set_local(q_backup.get_local())
        q_.vector().apply("insert")
        arc_solver.lmbda = lambda_backup

        if current_step_cutbacks >= max_failed_attempts:
            print("Too many failed attempts on this step. Stop.")
            break

        # Do not write output for a rejected attempt; retain it in the attempt count.
        continue

    # -------------------------------------------------
    # The attempted increment has converged; write accepted-state output only.
    # -------------------------------------------------
    converged_step += 1

    u_vis = project(u_, V_phi)
    displacement_in_load_direction = LOAD_DIRECTION_SIGN * u_vis(LOAD_POINT[0], LOAD_POINT[1])[2]
    actual_load = arc_solver.lmbda * P_REF * 4
    accepted_arc_length_by_ds0 = arc_solver.get_accepted_arc_length(normalized=True)

    if SAVE_DISPLACEMENT:
        u_vis.rename("u", "u")
        file_u << (u_vis, arc_solver.lmbda)

    # Append accepted states only to the data file.
    if SAVE_DAT:
        with open(dat_path, "a") as f:
            f.write(f"{actual_load:.12e}\t{displacement_in_load_direction:.12e}\n")

    print(
        f"attempt={attempt}, "
        f"converged_step={converged_step}, "
        f"lambda={arc_solver.lmbda:.6e}, "
        f"load={actual_load:.6e}, "
        f"displacement_in_load_direction={displacement_in_load_direction:.6e}, "
        f"iters={niter}, "
        f"accepted_arc_length={accepted_arc_length_by_ds0:.12f}*Ds0, "
        f"cutbacks_this_step={current_step_cutbacks}, "
        f"cutbacks_total={total_cutbacks}"
    )

    if (
        accepted_arc_length_by_ds0 + ACCEPTED_ARC_LENGTH_STOP_TOL
        >= END_ACCEPTED_ARC_LENGTH_BY_DS0
    ):
        print(
            f"Stop: accepted arc length reached "
            f"{accepted_arc_length_by_ds0:.12f}*Ds0 "
            f"(limit={END_ACCEPTED_ARC_LENGTH_BY_DS0:.12f}*Ds0)."
        )
        break

    # Reset the current-step cutback counter after acceptance.
    current_step_cutbacks = 0

t1 = time.process_time()
print(f"\nTotal iters = {total_iters}")
print(f"Total attempts = {attempt}")
print(f"Total converged steps = {converged_step}")
print(f"Total cutbacks = {total_cutbacks}")
print(f"Accepted arc length = {arc_solver.get_accepted_arc_length(normalized=True):.12f}*Ds0")
print(f'CPU time = {t1 - t0:.6f} s')
if SAVE_DAT:
    print(f"dat file written to: {dat_path}")
else:
    print("dat file saving disabled")
if SAVE_DISPLACEMENT:
    print(f"displacement pvd written to: {displacement_path}")
else:
    print("displacement field saving disabled")
