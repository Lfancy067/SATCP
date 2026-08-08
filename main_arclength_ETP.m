%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exact tangent predictor (ETP), i.e. the standard tangent predictor.
%
% At each newly accepted equilibrium state after the first step, the
% tangent stiffness is assembled and the unit reference-load response
% is obtained from
%
%                   K_T * u_hat = F_ext.
%
% The response is oriented using the preceding accepted increment and
% scaled by the target arc length in the same augmented metric used by
% the corrector. If a step is retried with a reduced arc length from the
% same accepted state, the previously computed tangent response is reused.
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
% fname = "input_Arch_semicircle-nelem50-unsym.txt";

outputDir = fullfile(pwd, 'data/ETP/');
solutionFileName = 'solution_exact_tangent_predictor.dat';
pathFileName = 'Path_exact_tangent_predictor.dat';
cpuSummaryFileName = 'cpu_time_exact_tangent_predictor.txt';

enablePlot = false;
enableDatOutput = false;
enableCpuSummaryOutput = false;
verbose = false;
showLoadStepProgress = true;

%% ------------------------------------------------------------------------
% Predictor and nonlinear-solver parameters
% -------------------------------------------------------------------------
maxNewtonIters = 10;
normTol = 1.0e-12;

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

% Augmented spherical arc-length metric:
% <z1,z2>_a = u1^T*u2 + psi*l1*l2*(Fext^T*Fext).
% solve_arclength_split.m uses the same value psi = 1.0.
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

% Exact unit reference-load response cached at the latest accepted state.
% A failed attempt changes only Ds, so the same response can be rescaled
% without repeating the tangent assembly, factorisation and linear solve.
exactTangentUnitResponse = zeros(neq,1);
exactTangentCacheAcceptedStep = -1;
hasExactTangentCache = false;

%% ------------------------------------------------------------------------
% Load-step attempts
% -------------------------------------------------------------------------
for loadStep = 1:maxloadSteps
    if(showLoadStepProgress)
        fprintf('\n==================== Load step %d / %d ====================\n', ...
                loadStep, maxloadSteps);
    end

    acceptedStepIndex = loadStepConverged + 1;
    isFirstAcceptedStep = (loadStepConverged == 0);

    if(isFirstAcceptedStep)
        % The first accepted step is solved by load control.
        disp = dispPrev;
        loadfactor = loadfactorPrev + Ds;

        if(verbose)
            fprintf("First-step load-control predictor: Dl = %12.6E\n", Ds);
        end
    else
        % Assemble and solve the exact tangent only once for each accepted
        % equilibrium point.  A reduced-Ds restart reuses this response.
        cacheValid = hasExactTangentCache ...
                  && (exactTangentCacheAcceptedStep == loadStepConverged);

        if(~cacheValid)
            Ktangent = assemble_tangent_stiffness( ...
                ndim, ndof, nelem, elemData, elemConn, coords, ...
                dispPrev, bf, LM, neq);

            exactTangentUnitResponse = solve_exact_unit_tangent_response( ...
                Ktangent, neq, assy4r, Fext);
            exactTangentCacheAcceptedStep = loadStepConverged;
            hasExactTangentCache = true;

            if(verbose)
                fprintf("Exact tangent response assembled at accepted step %d.\n", ...
                        loadStepConverged);
            end
        elseif(verbose)
            fprintf("Exact tangent response reused after step cutback.\n");
        end

        delta_u_n = dispPrev - dispPrev2;
        delta_l_n = loadfactorPrev - loadfactorPrev2;

        [delta_u_tangent, delta_l_tangent, signVal] = ...
            compute_exact_tangent_predictor( ...
                exactTangentUnitResponse, Ds, ...
                delta_u_n, delta_l_n, psi, FtF, normTol);

        disp = dispPrev + delta_u_tangent;
        loadfactor = loadfactorPrev + delta_l_tangent;

        if(verbose)
            fprintf("Exact tangent predictor sign: %4.0f\n", signVal);
            fprintf("Exact tangent predictor Dl: %12.6E\n", ...
                    delta_l_tangent);
        end
    end

    Du = disp - dispPrev;
    Dl = loadfactor - loadfactorPrev;

    convergedPrev = converged;
    converged = false;

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

        [converged, du, dl] = solve_arclength_split( ...
            acceptedStepIndex, neq, iter, Kglobal, Rglobal, ...
            dof_force, Fext, assy4r, Du, Dl, Ds);

        totalIters = totalIters + 1;

        if(converged)
            break;
        end

        disp(assy4r) = disp(assy4r) + du;
        loadfactor = loadfactor + dl;

        Du(assy4r) = Du(assy4r) + du;
        Dl = Dl + dl;
    end

    %% --------------------------------------------------------------------
    % Accept the solution or restart the step
    % ---------------------------------------------------------------------
    if(converged)
        if(acceptedStepIndex == 1)
            Ds = sqrt(Du' * Du + psi * Dl * Dl * FtF);
            DsMax = Ds;
            DsMin = Ds / 1024.0;
        end

        DsAccepted = Ds;

        loadfactorPrev2 = loadfactorPrev;
        loadfactorPrev  = loadfactor;

        dispPrev2 = dispPrev;
        dispPrev  = disp;

        DsPrev = DsAccepted;
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

        % The accepted base point has changed.  Its successor requires a
        % fresh exact tangent response; failed attempts leave the cache valid.
        hasExactTangentCache = false;
        exactTangentCacheAcceptedStep = -1;
    else
        countrestarts = countrestarts + 1;

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
% =========================================================================
function Ktangent = assemble_tangent_stiffness( ...
    ndim, ndof, nelem, elemData, elemConn, coords, dispState, bf, LM, neq)

    Ktangent = zeros(neq,neq);

    if(ndim == 2)
        if(ndof == 2)
            for e = 1:nelem
                [Klocal, ~] = Truss_2D_model1( ...
                    elemData, elemConn, e, coords, dispState, bf);
                Ktangent = Assembly_Matrix(Ktangent, Klocal, LM, e);
            end
        else
            for e = 1:nelem
                [Klocal, ~] = GeomExactBeam_2D( ...
                    elemData, elemConn, e, coords, dispState, bf);
                Ktangent = Assembly_Matrix(Ktangent, Klocal, LM, e);
            end
        end
    else
        if(ndof == 3)
            for e = 1:nelem
                [Klocal, ~] = Truss_3D_model2( ...
                    elemData, elemConn, e, coords, dispState, bf);
                Ktangent = Assembly_Matrix(Ktangent, Klocal, LM, e);
            end
        end
    end
end

function duUnitFull = solve_exact_unit_tangent_response( ...
    Ktangent, neq, assy4r, Fext)
    % Exact unit reference-load response at the accepted equilibrium point.
    K1 = Ktangent(assy4r, assy4r);
    FextReduced = Fext(assy4r);
    [L, U, P] = lu(sparse(K1));
    duu = L \ (P * FextReduced);
    duUnit = U \ duu;

    duUnitFull = zeros(neq,1);
    duUnitFull(assy4r) = duUnit;

    if(any(~isfinite(duUnitFull)))
        error('The exact unit tangent response contains non-finite values.');
    end
end

function [duTan, dlTan, signVal] = compute_exact_tangent_predictor( ...
    duUnitFull, Ds, delta_u_n, delta_l_n, psi, FtF, normTol)

    % Select the branch having a non-negative augmented inner product with
    % the preceding accepted increment.
    orientation = duUnitFull' * delta_u_n ...
                + psi * delta_l_n * FtF;
    if(orientation >= 0.0)
        signVal = 1.0;
    else
        signVal = -1.0;
    end

    tangentNormSquared = duUnitFull' * duUnitFull + psi * FtF;
    if(~isfinite(tangentNormSquared) || tangentNormSquared <= normTol^2)
        error('The exact tangent predictor has an invalid tangent norm.');
    end

    dlTan = signVal * Ds / sqrt(tangentNormSquared);
    duTan = dlTan * duUnitFull;
end
