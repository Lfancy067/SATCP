<!-- SPDX-FileCopyrightText: 2026 Fan Liu -->
<!-- SPDX-License-Identifier: MIT -->

# Licence information

This repository contains files distributed under different open-source licences.

## File-level licence mapping

| File or material | Licence and notice |
|---|---|
| `Present_solver.py` | MIT; Fan Liu notice in `LICENSES/MIT.txt` |
| `main_arclength_Present.m` | MIT; Kadapa upstream notice and Fan Liu modification notice |
| `solve_arclength_split.m` | MIT; Kadapa upstream notice and Fan Liu modification notice |
| `Example6.py` | GNU LGPL-3.0-or-later |
| `Example7.py` | GNU LGPL-3.0-or-later |
| `README.md`, `CITATION.cff`, `THIRD_PARTY_NOTICES.md` and `.gitignore` | MIT; Fan Liu notice in `LICENSES/MIT.txt` |

The complete standard licence texts and copyright notices are located in:

```text
LICENSES/MIT.txt
LICENSES/ArcLengthMethod-MIT.txt
LICENSES/LGPL-3.0-or-later.txt
LICENSES/GPL-3.0-or-later.txt
```

`LICENSES/MIT.txt` covers the original MIT-licensed material authored by Fan Liu. `LICENSES/ArcLengthMethod-MIT.txt` reproduces the upstream MIT notice of Chennakesava Kadapa's `ArcLengthMethod` project. The modified MATLAB files retain both copyright notices in their file headers.

The LGPL version 3 incorporates the terms of GPL version 3 and adds further permissions; both standard texts are included.

## File headers

Each source file retains its own SPDX licence identifier and copyright information. Where a file is adapted from an upstream project, its header identifies the upstream source and the modifications.

If this document conflicts with an explicit file-level SPDX identifier or an upstream copyright notice, the file-level information governs that file.

## Third-party software

The upstream `ArcLengthMethod`, legacy FEniCS/DOLFIN and FEniCS-Shells projects retain their own copyrights and licences. See `THIRD_PARTY_NOTICES.md`.
