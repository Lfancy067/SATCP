% SPDX-FileCopyrightText: 2020 Chennakesava Kadapa
% SPDX-FileCopyrightText: 2026 Fan Liu
%
% SPDX-License-Identifier: MIT
%
% Upstream MIT notice: LICENSES/ArcLengthMethod-MIT.txt
% Modification MIT notice: LICENSES/MIT.txt
%
% Based on solve_arclength_split.m from:
% https://github.com/chennachaos/ArcLengthMethod
%
% Modified by Fan Liu in 2026 for MATLAB compatibility and SATCP:
% - return the unit reference-load response du1 used by a correction;
% - initialise du1 when the trial state is already converged;
% - check convergence before factorisation and correction;
% - expose the arc metric and numerical tolerances as optional inputs;
% - reject non-finite solves and unsafe split-equation denominators.

function [converged, du, dl, du1, correctionValid] = ...
    solve_arclength_split(timeStep, ~, iter, Kglobal, Rglobal, ...
    ~, Fext, assy4r, Du, Dl, ds, psi, absTol, denominatorTol)

    if nargin < 12 || isempty(psi)
        psi = 1.0;
    end
    if nargin < 13 || isempty(absTol)
        absTol = 1.0e-6;
    end
    if nargin < 14 || isempty(denominatorTol)
        denominatorTol = 1.0e-14;
    end

    FextReduced = Fext(assy4r);
    FtF = Fext' * Fext;

    if timeStep > 1
        A = Du' * Du + psi * Dl * Dl * FtF - ds * ds;
        a = 2.0 * Du(assy4r)';
        b = 2.0 * psi * Dl * FtF;
    else
        % First accepted increment: prescribed-load Newton step.
        A = 0.0;
        a = zeros(1, numel(assy4r));
        b = 1.0;
    end

    R = Rglobal(assy4r);
    residualNorm = norm(R, 2);
    combinedNorm = sqrt(residualNorm * residualNorm + A * A);

    fprintf(' rNorm : %5d ...  %12.6E \n', iter, combinedNorm);

    du = zeros(size(R));
    dl = 0.0;
    du1 = [];
    converged = false;
    correctionValid = true;

    if ~isfinite(combinedNorm)
        correctionValid = false;
        return;
    end

    % A converged trial state requires no additional Newton correction.
    if combinedNorm < absTol
        converged = true;
        return;
    end

    K1 = Kglobal(assy4r, assy4r);
    [L, U, P] = lu(sparse(K1));

    rhs = L \ (P * FextReduced);
    du1 = U \ rhs;

    rhs = L \ (P * R);
    du2 = -(U \ rhs);

    if any(~isfinite(du1)) || any(~isfinite(du2))
        du1 = [];
        correctionValid = false;
        return;
    end

    denominator = b + a * du1;
    denominatorScale = max([1.0, abs(b), norm(a, 2) * norm(du1, 2)]);
    if ~isfinite(denominator) || ...
       abs(denominator) <= denominatorTol * denominatorScale
        du1 = [];
        correctionValid = false;
        return;
    end

    dl = (a * du2 - A) / denominator;
    du = -du2 + dl * du1;

    if ~isfinite(dl) || any(~isfinite(du))
        du = zeros(size(R));
        dl = 0.0;
        du1 = [];
        correctionValid = false;
    end
end
