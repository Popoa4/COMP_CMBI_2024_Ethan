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
fprintf('Method 0 (random initial point) time consumption: %.2f s\n', time_origin);

%% exp1
[results_rician, time_rician] = method_rician(dwis, bvals, qhat, slice_num);
visualize_results(results_rician, 'Method with rician noise'); 
fprintf('Method with rician noise time consumption: %.2f s\n', time_rician);

%% Print the average RESNORM value of each method
avg_RESNORM_origin = mean(results_origin.RESNORM(~isnan(results_origin.RESNORM)));
avg_RESNORM_rician = mean(results_rician.RESNORM(~isnan(results_rician.RESNORM)));

fprintf('\n===== Method performance comparison =====\n');
fprintf('Method 0 average RESNORM: %.4e\n', avg_RESNORM_origin);
fprintf('Method 1 average RESNORM: %.4e\n', avg_RESNORM_rician);