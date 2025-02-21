% Comparison of three methods
% 0. Using set initial point + parameter transformation + fminunc
% 1. Using DTI-based initial point + parameter transformation + fminunc
% 2. Using DTI initial point + fmincon (constrained optimization)
% 3. Using DTI initial point + fmincon + analytical gradient
%% Data loading
clear; clc; close all;
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);
slice_num = 72;

%% exp0
[results_origin, time_origin] = method_origin(dwis, bvals, qhat, slice_num);
visualize_results(results_origin, 'Origin Initialization Method');
fprintf('\n===== Method comparison =====\n');
fprintf('Method 0 (random initial point) time: %.2f s\n', time_origin);

%% exp1
[results_dti, time_dti] = method_dti_initial(dwis, bvals, qhat, slice_num);
visualize_results(results_dti, 'DTI Initialization Method');
fprintf('Method 1 (DTI initial point) time consumption: %.2f s\n', time_dti);

%% exp2
[results_fmincon, time_fmincon] = method_fmincon(dwis, bvals, qhat, slice_num);
visualize_results(results_fmincon, 'fmincon Constrained Method');
fprintf('Method 2 (fmincon constraint) time consumption: %.2f s\n', time_fmincon);

%% exp3
[results_analytic, time_analytic] = method_analytic_grad(dwis, bvals, qhat, slice_num);
visualize_results(results_analytic, 'Analytic Gradient Method');
fprintf('Method 3 (analytic gradient) time consumption: %.2f s\n', time_analytic);

%% Print the average RESNORM value of each method
avg_RESNORM_origin = mean(results_origin.RESNORM(~isnan(results_origin.RESNORM)));
avg_RESNORM_dti = mean(results_dti.RESNORM(~isnan(results_dti.RESNORM)));
avg_RESNORM_fmincon = mean(results_fmincon.RESNORM(~isnan(results_fmincon.RESNORM)));
avg_RESNORM_analytic = mean(results_analytic.RESNORM(~isnan(results_analytic.RESNORM)));
fprintf('\n===== Method performance comparison =====\n');
fprintf('Method 0 average RESNORM: %.4e\n', avg_RESNORM_origin);
fprintf('Method 1 average RESNORM: %.4e\n', avg_RESNORM_dti);
fprintf('Method 2 average RESNORM: %.4e\n', avg_RESNORM_fmincon);
fprintf('Method 3 average RESNORM: %.4e\n', avg_RESNORM_analytic);