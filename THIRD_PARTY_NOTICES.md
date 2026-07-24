<!-- SPDX-FileCopyrightText: 2026 Fan Liu -->
<!-- SPDX-License-Identifier: MIT -->

# Third-party notices

This file records the principal third-party software and source material used by the released code.

## Chennakesava Kadapa — ArcLengthMethod

`main_arclength_Present.m` is designed to operate within Chennakesava Kadapa's open-source MATLAB implementation of the arc-length method:

- Project: `ArcLengthMethod`
- Upstream repository: https://github.com/chennachaos/ArcLengthMethod
- Compatibility-tested commit: `b250d639ce7bd497d568eb262b8df6cb723e273c`
- Upstream licence: MIT
- Upstream MIT notice: `LICENSES/ArcLengthMethod-MIT.txt`

The upstream framework supplies finite element assembly routines, element formulations, benchmark input files and the baseline split arc-length corrector.

The complete upstream framework is not redistributed in this repository. Users should obtain it directly from the upstream repository and then copy the released `main_arclength_Present.m` and `solve_arclength_split.m` into the corresponding MATLAB working directory.

`main_arclength_Present.m` and `solve_arclength_split.m` modify upstream MIT-licensed files. Their source headers retain Chennakesava Kadapa's copyright together with the notice covering Fan Liu's modifications. The released files implement SATCP, expose the unit reference-load response required by the approximate-tangent history, and add trial/commit and numerical-validity safeguards.

## FEniCS-Shells

`Example6.py` and `Example7.py` are adapted from the nonlinear Naghdi cylindrical-shell demonstration of the FEniCS-Shells project:

- Demo: `demo_nonlinear-naghdi-cylindrical.py`
- Demo documentation: https://fenics-shells.readthedocs.io/en/latest/demo/nonlinear-naghdi-cylindrical/demo_nonlinear-naghdi-cylindrical.py.html
- Project repository: https://bitbucket.org/unilucompmech/fenics-shells
- Upstream licence: GNU LGPL-3.0-or-later

The adapted portions include the nonlinear Naghdi shell kinematics, mixed finite element interpolation and partial selective reduced integration framework.

The modifications by Fan Liu include benchmark-specific geometry, mesh and boundary conditions, concentrated-load treatment, coupling with the SATCP arc-length solver, continuation controls, restart handling and accepted-state output. The modified benchmark files remain distributed under LGPL-3.0-or-later.

Relevant reference:

J. S. Hale, M. Brunetti, S. P. A. Bordas, and C. Maurini, “Simple and extensible plate and shell finite element models through automatic code generation tools”, *Computers & Structures*, 209 (2018), 163–181.

## Legacy FEniCS/DOLFIN

The Python files use the public interface of legacy FEniCS/DOLFIN. FEniCS/DOLFIN is not redistributed by this repository and must be installed separately.

Relevant reference:

M. S. Alnæs et al., “The FEniCS Project Version 1.5”, *Archive of Numerical Software*, 3 (2015).
