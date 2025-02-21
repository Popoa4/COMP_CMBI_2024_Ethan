clear; clc; close all;
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]); 
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);
slice_num = 72;
    
%% Selecting a single voxel
voxel_coords = [92, 65, 72];
% voxel_coords = [90, 60, 70];
% voxel_coords = [92, 63, 70];

Y = build_design_matrix(bvals, qhat); 
[ci_95_MCMC, total_time] = mcmc_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y);

    
fprintf('MCMC 95%% confidence intervals for parameters:\n');
fprintf('S0: [%.2f, %.2f]\n', ci_95_MCMC(1,1), ci_95_MCMC(2,1));
fprintf('d : [%.6f, %.6f]\n', ci_95_MCMC(1,2), ci_95_MCMC(2,2));
fprintf('f : [%.4f, %.4f]\n', ci_95_MCMC(1,3), ci_95_MCMC(2,3));
    