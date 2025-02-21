clear; clc; close all;
load('data');  
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]); 
load('bvecs');    
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2); 

slice_num = 72;

% Selecting a single voxel
voxel_coords = [92, 65, 72];
Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
Y = build_design_matrix(bvals, qhat);

% Running different uncertainty estimation methods
% 1. Classic bootstrap
[ci_bootstrap, time_bootstrap] = bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, 1000);

% 2. Bootstrap for parameters
[ci_param_bootstrap, time_param_bootstrap] = parametric_bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, 1000);

% 3. Laplace method
[ci_laplace, time_laplace] = laplace_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y);

% 4. MCMC
[ci_mcmc, time_mcmc] = mcmc_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y);


% Comparison of results
fprintf('\n===== Confidence intervals for parameters from different methods =====\n');
fprintf('Method\t\tS0 confidence intervals\n');
fprintf('Bootstrap\t[%.2f, %.2f]\n', ci_bootstrap(1,1), ci_bootstrap(2,1));
fprintf('Param Bootstrap\t[%.2f, %.2f]\n', ci_param_bootstrap(1,1), ci_param_bootstrap(2,1));
fprintf('Laplace\t\t[%.2f, %.2f]\n', ci_laplace(1,1), ci_laplace(1,2));
fprintf('MCMC\t\t[%.2f, %.2f]\n', ci_mcmc(1,1), ci_mcmc(2,1));

% 显示各方法的计算时间
fprintf('\n===== Calculation time =====\n');
fprintf('Bootstrap: %.2f s\n', time_bootstrap);
fprintf('Param Bootstrap: %.2f s\n', time_param_bootstrap);
fprintf('Laplace: %.2f s\n', time_laplace);
fprintf('MCMC: %.2f s\n', time_mcmc);
