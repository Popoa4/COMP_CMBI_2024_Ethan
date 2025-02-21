clear; clc; close all;

%% Data loading and preprocessing
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);
slice_num = 72;
    
%% Selecting a single voxel
% voxel_coords = [92, 65, 72];
% voxel_coords = [90, 60, 70];
voxel_coords = [92, 63, 70];
    
Y = build_design_matrix(bvals, qhat);
    
%% Bootstrapping
[ci_2sigma, time_bootstrap] = bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, 1000);

%% Computing confidence intervals
fprintf('Bootstrap 95% confidence intervals:\n');
fprintf('S0: [%.2f, %.2f]\n', ci_2sigma(1,1), ci_2sigma(2,1));
fprintf('d : [%.6f, %.6f]\n', ci_2sigma(1,2), ci_2sigma(2,2));
fprintf('f : [%.4f, %.4f]\n', ci_2sigma(1,3), ci_2sigma(2,3));