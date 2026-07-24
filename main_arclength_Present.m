% SPDX-FileCopyrightText: 2020 Chennakesava Kadapa
% SPDX-FileCopyrightText: 2026 Fan Liu
%
% SPDX-License-Identifier: MIT
%
% Upstream MIT notice: LICENSES/ArcLengthMethod-MIT.txt
% Modification MIT notice: LICENSES/MIT.txt
%
% This file is based on the ArcLengthMethod code by Chennakesava Kadapa:
% https://github.com/chennachaos/ArcLengthMethod
%
% Modified by Fan Liu in 2026.
% Major modifications:
% - implementation of the SATCP predictor;
% - curvature and approximate-tangent history;
% - restart-safe trial/commit procedure;
% - revised continuation and output controls.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Proposed secant-approximate-tangent curvature predictor (Present).
%
% Main ideas:
%   1. Solve the first accepted step by load control.
%   2. Reuse the unit reference-load displacement response from the last
%      Newton correction of the preceding accepted step.
%   3. Construct the accepted secant and endpoint approximate tangent in
%      the augmented arc-length metric.
%   4. Estimate the local effective curvature from their included angle.
%   5. Predict the next secant direction in the local turning plane and
%      scale it by the target arc length.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
clc;
more off;
format long;
cpuStart = cputime;

%% ------------------------------------------------------------------------
% Model selection
% -------------------------------------------------------------------------
% fname = "input_Truss_2D_3members_model1.txt";
% fname = "input_Truss_3D_2members.txt";
% fname = "input_Truss_3D_12members.txt";
% fname = "input_LeeFrame-nelem20.txt";
% fname = "input_arch-215deg.txt";
% fname = "input_Arch_semicircle-nelem50-sym.txt";
fname = "input_Arch_semicircle-nelem50-unsym.txt";

outputDir = fullfile(pwd, 'data/present/');
solutionFileName = 'solution_present.dat';
pathFileName = 'Path_present.dat';
cpuSummaryFileName = 'cpu_time_present.txt';

% Disable plotting, data-file output and detailed console output for timing.
enablePlot = false;
enableDatOutput = false;
enableCpuSummaryOutput = false;
verbose = false;

% Print a separator before the Newton iterations of every attempted load
% step. This progress output is independent of the detailed verbose output.
showLoadStepProgress = true;

%% ------------------------------------------------------------------------
% Predictor and nonlinear-solver parameters
% -------------------------------------------------------------------------
maxNewtonIters = 10;

% Numerical safeguards for direction normalisation and angle evaluation.
normTol = 1.0e-12;
angleTol = 1.0e-8;

% Maximum predicted turning angle, phi_max = 20 degrees.
phiMax = 20.0 * pi / 180.0;

% Corrector convergence and split-equation safeguards.
absTol = 1.0e-6;
denominatorTol = 1.0e-14;

%% ------------------------------------------------------------------------
% Read model and initialise finite-element variables
% -------------------------------------------------------------------------
[ndim, ndof, nnode, nelem, coords, elemConn, elemData, LM, neq, assy4r, ...
 dof_force, Fext, maxloadSteps, loadincr, outputlist] = processfile(fname);

% The variable name "disp" is retained for compatibility with the existing
% element and assembly routines.
disp = zeros(neq,1);
dispPrev  = disp;
dispPrev2 = disp;

Kglobal = zeros(neq,neq);
Rglobal = zeros(neq,1);
bf = [0.0 0.0];

% Augmented arc-length metric:
% <z1,z2>_a = u1^T*u2 + psi*l1*l2*(Fext^T*Fext).
psi = 1.0;
FtF = Fext' * Fext;

Ds = loadincr;
DsPrev = Ds;
DsMax = Ds;
DsMin = Ds;

loadfactor      = loadincr;
loadfactorPrev  = 0.0;
loadfactorPrev2 = 0.0;

converged = false;
convergedPrev = false;
loadStepConverged = 0;

output = disp(outputlist);
llist = 0.0;
dispFull = disp;

countrestarts = 0;
totalIters = 0;

%% ------------------------------------------------------------------------
% Predictor history retained only from accepted states
% -------------------------------------------------------------------------
% Unit reference-load displacement response from the preceding accepted step.
du1LastFull = zeros(neq,1);
hasDu1Last = false;

% Effective curvature at the preceding accepted endpoint.
kappaPrev = 0.0;
hasKappaPrev = false;

% Latest accepted augmented increment and its unit direction.
hasDeltaUN = false;
delta_u_n = zeros(neq,1);
delta_l_n = 0.0;
delta_u_n_bar = zeros(neq,1);
delta_l_n_bar = 0.0;

%% ------------------------------------------------------------------------
% Load-step attempts
% -------------------------------------------------------------------------
for loadStep = 1:maxloadSteps
    if(showLoadStepProgress)
        fprintf('\n==================== Load step %d / %d ====================\n', ...
                loadStep, maxloadSteps);
    end

    % Trial predictor history is committed only after convergence.
    kappaTrial = kappaPrev;
    hasKappaTrial = false;

    acceptedStepIndex = loadStepConverged + 1;
    isFirstAcceptedStep = (loadStepConverged == 0);

    if(isFirstAcceptedStep)
        % Prescribed load increment for the first accepted step.
        disp = dispPrev;
        loadfactor = loadfactorPrev + Ds;

        if(verbose)
            fprintf("First-step load-control predictor: Dl = %12.6E\n", Ds);
        end

    elseif(~hasDeltaUN || ~hasDu1Last)
        % Start-up fallback. Under normal operation the required Present
        % history is available immediately after the first accepted step.
        alpha = Ds / DsPrev;
        disp = (1.0 + alpha) * dispPrev - alpha * dispPrev2;
        loadfactor = (1.0 + alpha) * loadfactorPrev ...
                   - alpha * loadfactorPrev2;

        if(verbose)
            fprintf("Start-up SEP fallback: alpha = %12.6E\n", alpha);
        end

    else
        % Proposed Present predictor.
        [disp, loadfactor, kappaTrial, hasKappaTrial, predictorInfo] = ...
            present_predictor( ...
                dispPrev, loadfactorPrev, Ds, DsPrev, psi, FtF, ...
                delta_u_n, delta_l_n, ...
                delta_u_n_bar, delta_l_n_bar, ...
                du1LastFull, kappaPrev, hasKappaPrev, ...
                phiMax, normTol, angleTol);

        if(verbose)
            fprintf(['Predictor mode: %s, theta = %12.6E, ', ...
                     'kappa = %12.6E, kappa_s = %12.6E, ', ...
                     'phi_raw = %12.6E, phi_used = %12.6E, ', ...
                     'phi_max = %12.6E\n'], ...
                     predictorInfo.mode, predictorInfo.theta, ...
                     predictorInfo.kappaNow, predictorInfo.kappaRate, ...
                     predictorInfo.phiTargetRaw, ...
                     predictorInfo.phiTarget, ...
                     predictorInfo.phiMax);
        end
    end

    Du = disp - dispPrev;
    Dl = loadfactor - loadfactorPrev;

    convergedPrev = converged;
    converged = false;

    % Trial unit-load response; commit only after the attempted step converges.
    du1StepFull = du1LastFull;
    hasDu1Step = false;

    %% --------------------------------------------------------------------
    % Newton corrector
    % ---------------------------------------------------------------------
    for iter = 1:maxNewtonIters
        Kglobal(1:end,1:end) = 0.0;
        Rglobal(1:end) = 0.0;

        if(ndim == 2)
            if(ndof == 2)
                for e = 1:nelem
                    [Klocal, Flocal] = Truss_2D_model1( ...
                        elemData, elemConn, e, coords, disp, bf);
                    Kglobal = Assembly_Matrix(Kglobal, Klocal, LM, e);
                    Rglobal = Assembly_Vector(Rglobal, Flocal, LM, e);
                end
            else
                for e = 1:nelem
                    [Klocal, Flocal] = GeomExactBeam_2D( ...
                        elemData, elemConn, e, coords, disp, bf);
                    Kglobal = Assembly_Matrix(Kglobal, Klocal, LM, e);
                    Rglobal = Assembly_Vector(Rglobal, Flocal, LM, e);
                end
            end
        else
            if(ndof == 3)
                for e = 1:nelem
                    [Klocal, Flocal] = Truss_3D_model2( ...
                        elemData, elemConn, e, coords, disp, bf);
                    Kglobal = Assembly_Matrix(Kglobal, Klocal, LM, e);
                    Rglobal = Assembly_Vector(Rglobal, Flocal, LM, e);
                end
            end
        end

        Rglobal = Rglobal + loadfactor * Fext;

        [converged, du, dl, du1, correctionValid] = ...
            solve_arclength_split( ...
            acceptedStepIndex, neq, iter, Kglobal, Rglobal, ...
            dof_force, Fext, assy4r, Du, Dl, Ds, ...
            psi, absTol, denominatorTol);

        totalIters = totalIters + 1;

        if(~correctionValid)
            if(verbose)
                fprintf(['Invalid linear solve or split-equation ', ...
                         'denominator; reject this attempted step.\n']);
            end
            break;
        end

        % A converged iterate requires no additional Newton correction.
        if(converged)
            break;
        end

        % Store only the unit-load response used in an actual correction.
        if(~isempty(du1))
            du1StepFull = zeros(neq,1);
            du1StepFull(assy4r) = du1;
            hasDu1Step = true;
        end

        % Apply the Newton correction.
        disp(assy4r) = disp(assy4r) + du;
        loadfactor = loadfactor + dl;

        Du(assy4r) = Du(assy4r) + du;
        Dl = Dl + dl;
    end

    %% --------------------------------------------------------------------
    % Accept the solution or restart the step
    % ---------------------------------------------------------------------
    if(converged)
        % Establish the arc-length scale from the first accepted step.
        if(acceptedStepIndex == 1)
            Ds = sqrt(Du' * Du + psi * Dl * Dl * FtF);
            DsMax = Ds;
            DsMin = Ds / 1024.0;
        end

        DsAccepted = Ds;

        % Commit the endpoint approximate-tangent information.
        if(hasDu1Step)
            du1LastFull = du1StepFull;
            hasDu1Last = true;
        end

        % Commit the predictor curvature only after convergence.
        if(hasKappaTrial)
            kappaPrev = kappaTrial;
            hasKappaPrev = true;
        end

        % Update the latest accepted augmented increment.
        [delta_u_n, delta_l_n, delta_u_n_bar, delta_l_n_bar, ...
         hasDeltaUN] = update_latest_increment( ...
            disp, loadfactor, dispPrev, loadfactorPrev, ...
            psi, FtF, normTol);

        % Shift only the accepted states required by the start-up fallback.
        loadfactorPrev2 = loadfactorPrev;
        loadfactorPrev  = loadfactor;

        dispPrev2 = dispPrev;
        dispPrev  = disp;

        DsPrev = DsAccepted;

        % Preserve the original adaptive arc-length rule.
        if(convergedPrev)
            Ds = min(max(2.0 * Ds, DsMin), DsMax);
        end

        if(enableDatOutput || enablePlot)
            dispFull = [dispFull; disp]; %#ok<AGROW>
            output = [output disp(outputlist)]; %#ok<AGROW>
            llist = [llist; loadfactor]; %#ok<AGROW>
        end

        if(enablePlot)
            plot(abs(output(2,:)), llist, 'bs-');
            hold on;
        end

        loadStepConverged = loadStepConverged + 1;

    else
        countrestarts = countrestarts + 1;

        % Failed attempts do not modify accepted predictor history.
        if(convergedPrev)
            Ds = max(Ds * 0.5, DsMin);
        else
            Ds = max(Ds * 0.25, DsMin);
        end

        if(verbose)
            fprintf("Restart: new arc length Ds = %12.6E\n", Ds);
        end
    end
end

%% ------------------------------------------------------------------------
% Output summary
% -------------------------------------------------------------------------
fprintf('Restart count = %d\n', countrestarts);
fprintf('Total nonlinear iterations = %d\n', totalIters);
fprintf('Converged load steps = %d\n', loadStepConverged);

if((enableDatOutput || enableCpuSummaryOutput) && ~exist(outputDir, 'dir'))
    mkdir(outputDir);
end

if(enableDatOutput)
    fileID = fopen(fullfile(outputDir, solutionFileName), 'w');
    if(fileID < 0)
        error('Unable to open the solution output file.');
    end
    for ii = 1:size(dispFull,1)
        fprintf(fileID, '%12.8f \n', dispFull(ii));
    end
    fclose(fileID);

    fileID = fopen(fullfile(outputDir, pathFileName), 'w');
    if(fileID < 0)
        error('Unable to open the path output file.');
    end
    for ii = 1:size(llist,1)
        fprintf(fileID, '%12.8f \t %12.8f \t %12.8f \n', ...
            llist(ii), output(1,ii), output(2,ii));
    end
    fclose(fileID);
end

cpuTime = cputime - cpuStart;
fprintf('CPU time = %12.6f seconds\n', cpuTime);

if(enableCpuSummaryOutput)
    fileID = fopen(fullfile(outputDir, cpuSummaryFileName), 'w');
    if(fileID < 0)
        error('Unable to open the CPU summary output file.');
    end
    fprintf(fileID, 'input_file: %s\n', fname);
    fprintf(fileID, 'solution_file: %s\n', solutionFileName);
    fprintf(fileID, 'path_file: %s\n', pathFileName);
    fprintf(fileID, 'cpu_time_seconds: %12.6f\n', cpuTime);
    fprintf(fileID, 'total_nonlinear_iterations: %d\n', totalIters);
    fprintf(fileID, 'converged_load_steps: %d\n', loadStepConverged);
    fprintf(fileID, 'restart_count: %d\n', countrestarts);
    fclose(fileID);
end

%% ========================================================================
% Local functions
% ========================================================================
function val = arc_inner(u1, l1, u2, l2, psi, FtF)
    val = u1' * u2 + psi * l1 * l2 * FtF;
end

function [uBar, lBar, valid] = standardize_increment( ...
    u, l, psi, FtF, normTol)

    normSquared = arc_inner(u, l, u, l, psi, FtF);
    valid = isfinite(normSquared) && (normSquared > normTol^2);

    if(valid)
        incrementNorm = sqrt(normSquared);
        uBar = u / incrementNorm;
        lBar = l / incrementNorm;
    else
        uBar = zeros(size(u));
        lBar = 0.0;
    end
end

function value = clamp(value, lowerBound, upperBound)
    value = min(upperBound, max(lowerBound, value));
end

function [duTanBar, dlTanBar, valid] = compute_tangent_direction( ...
    du1LastFull, delta_u_n, delta_l_n, psi, FtF, normTol)

    % Reference approximate-tangent direction: [du1LastFull; 1].
    orientation = delta_u_n' * du1LastFull ...
                + psi * delta_l_n * FtF;

    if(orientation >= 0.0)
        signVal = 1.0;
    else
        signVal = -1.0;
    end

    tangentNormSquared = du1LastFull' * du1LastFull + psi * FtF;
    valid = isfinite(tangentNormSquared) ...
         && (tangentNormSquared > normTol^2);

    if(valid)
        tangentNorm = sqrt(tangentNormSquared);
        duTanBar = signVal * du1LastFull / tangentNorm;
        dlTanBar = signVal / tangentNorm;
    else
        duTanBar = zeros(size(du1LastFull));
        dlTanBar = 0.0;
    end
end

function [uPredBar, lPredBar, kappaNow, kappaRate, ...
          phiTarget, phiTargetRaw, phiClippedByMax, theta, valid] = ...
        present_direction( ...
        uNBar, lNBar, uTanBar, lTanBar, DsPrev, DsNext, ...
        kappaPrev, hasKappaPrev, phiMax, psi, FtF, normTol, angleTol)

    % Effective curvature from the accepted secant and endpoint tangent.
    cosNT = arc_inner(uNBar, lNBar, uTanBar, lTanBar, psi, FtF);
    cosNT = clamp(cosNT, -1.0, 1.0);
    theta = acos(cosNT);
    kappaNow = 2.0 * theta / max(DsPrev, normTol);

    if(hasKappaPrev)
        kappaRate = (kappaNow - kappaPrev) / max(DsPrev, normTol);
    else
        % First prediction uses the constant-curvature contribution only.
        kappaRate = 0.0;
    end

    % Predict the next secant turning angle and apply its bounds.
    phiTargetRaw = 0.5 * kappaNow * DsNext ...
                 + (1.0 / 6.0) * kappaRate * DsNext^2;
    phiTarget = max(0.0, phiTargetRaw);
    phiClippedByMax = false;

    if(isfinite(phiMax) && phiTarget > phiMax)
        phiTarget = phiMax;
        phiClippedByMax = true;
    end

    % The turning plane is undefined in the limiting collinear cases.
    if(theta <= angleTol || theta >= pi - angleTol || ...
       phiTarget <= angleTol)
        uPredBar = uTanBar;
        lPredBar = lTanBar;
        valid = true;
        return;
    end

    % Forward unit direction in the local secant-tangent turning plane:
    % q_n = cos(theta)*q_T - sin(theta)*v_forward.
    sinTheta = sin(theta);
    uForward = (cosNT * uTanBar - uNBar) / sinTheta;
    lForward = (cosNT * lTanBar - lNBar) / sinTheta;

    [uForward, lForward, forwardValid] = standardize_increment( ...
        uForward, lForward, psi, FtF, normTol);

    if(~forwardValid)
        uPredBar = uTanBar;
        lPredBar = lTanBar;
        valid = true;
        return;
    end

    uPred = cos(phiTarget) * uTanBar + sin(phiTarget) * uForward;
    lPred = cos(phiTarget) * lTanBar + sin(phiTarget) * lForward;

    [uPredBar, lPredBar, valid] = standardize_increment( ...
        uPred, lPred, psi, FtF, normTol);

    if(~valid)
        uPredBar = uTanBar;
        lPredBar = lTanBar;
        valid = true;
    end
end

function [dispPred, loadPred, kappaNow, hasKappaNow, info] = ...
    present_predictor( ...
        dispPrev, loadfactorPrev, Ds, DsPrev, psi, FtF, ...
        delta_u_n, delta_l_n, delta_u_n_bar, delta_l_n_bar, ...
        du1LastFull, kappaPrev, hasKappaPrev, phiMax, normTol, angleTol)

    info.mode = 'Present';
    info.theta = NaN;
    info.kappaNow = NaN;
    info.kappaRate = NaN;
    info.phiTargetRaw = NaN;
    info.phiTarget = NaN;
    info.phiMax = phiMax;
    info.phiClippedByMax = false;

    % Approximate tangent at the current accepted equilibrium point.
    [uTanBar, lTanBar, tangentValid] = compute_tangent_direction( ...
        du1LastFull, delta_u_n, delta_l_n, psi, FtF, normTol);

    if(~tangentValid)
        error('Approximate tangent direction is invalid.');
    end

    [uPredBar, lPredBar, kappaNow, kappaRate, ...
     phiTarget, phiTargetRaw, phiClippedByMax, theta, curvatureValid] = ...
        present_direction( ...
            delta_u_n_bar, delta_l_n_bar, ...
            uTanBar, lTanBar, DsPrev, Ds, ...
            kappaPrev, hasKappaPrev, phiMax, psi, FtF, normTol, angleTol);

    if(~curvatureValid)
        uPredBar = uTanBar;
        lPredBar = lTanBar;
        info.mode = 'tangent fallback';
    elseif(theta <= angleTol || phiTarget <= angleTol)
        info.mode = 'tangent limit';
    elseif(phiClippedByMax)
        info.mode = 'phi_max clipped';
    end

    dispPred = dispPrev + Ds * uPredBar;
    loadPred = loadfactorPrev + Ds * lPredBar;

    info.theta = theta;
    info.kappaNow = kappaNow;
    info.kappaRate = kappaRate;
    info.phiTargetRaw = phiTargetRaw;
    info.phiTarget = phiTarget;
    info.phiClippedByMax = phiClippedByMax;
    hasKappaNow = true;
end

function [delta_u_n, delta_l_n, delta_u_n_bar, delta_l_n_bar, ...
          hasDeltaUN] = update_latest_increment( ...
        dispNew, loadNew, dispOld, loadOld, psi, FtF, normTol)

    delta_u_n = dispNew - dispOld;
    delta_l_n = loadNew - loadOld;

    [delta_u_n_bar, delta_l_n_bar, valid] = standardize_increment( ...
        delta_u_n, delta_l_n, psi, FtF, normTol);

    if(~valid)
        error('The newly accepted increment has zero or invalid arc norm.');
    end

    hasDeltaUN = true;
end
