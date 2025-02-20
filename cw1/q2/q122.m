%% 数据加载和预处理（复用前面的问题代码）
clear; clc; close all;
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]); % 调整维度为 [108,145,174,145]
    
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);
slice_num = 72;
    
%% 选择单个体素 (92, 65, 72)
% voxel_coords = [92, 65, 72];
% voxel_coords = [90, 60, 70];
voxel_coords = [92, 63, 70];
% Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    
%% 构建设计矩阵 Y (复用Q1.1.1)
Y = build_design_matrix(bvals, qhat);
    
[ci_95_MCMC, total_time] = mcmc_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y);

    
fprintf('参数MCMC 95%%置信区间:\n');
fprintf('S0: [%.2f, %.2f]\n', ci_95_MCMC(1,1), ci_95_MCMC(2,1));
fprintf('d : [%.6f, %.6f]\n', ci_95_MCMC(1,2), ci_95_MCMC(2,2));
fprintf('f : [%.4f, %.4f]\n', ci_95_MCMC(1,3), ci_95_MCMC(2,3));
    
%% 可视化采样结果
% figure;
% subplot(3,1,1);
% histogram(samples_post(:,1), 50); title('S0 Posterior Distribution');
% subplot(3,1,2);
% histogram(samples_post(:,2), 50); title('d Posterior Distribution');
% subplot(3,1,3);
% histogram(samples_post(:,3), 50); title('f Posterior Distribution');
    