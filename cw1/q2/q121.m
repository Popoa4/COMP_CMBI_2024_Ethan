% 清空环境
clear; clc; close all;

%% 数据加载和预处理
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]); % 调整维度为 [108,145,174,145]
    
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);
slice_num = 72;
    
%% 选择单个体素 (92, 65, 72)
voxel_coords = [92, 65, 72];
% Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    
%% 构建设计矩阵 Y
Y = build_design_matrix(bvals, qhat);
    
%% 计算DTI参数作为初始点
% [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    
%% 自助法
[ci_2sigma, time_bootstrap] = bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, 1000);

%% 计算置信区间
% ci_2sigma = prctile(params_bootstrap, [2.5 97.5]); % 95%置信区间
% ci_95 = prctile(params_bootstrap, [2.5 97.5]);      % 与2-sigma相同
    
% 输出结果
fprintf('参数Bootstrap 95%%置信区间:\n');
fprintf('S0: [%.2f, %.2f]\n', ci_2sigma(1,1), ci_2sigma(2,1));
fprintf('d : [%.6f, %.6f]\n', ci_2sigma(1,2), ci_2sigma(2,2));
fprintf('f : [%.4f, %.4f]\n', ci_2sigma(1,3), ci_2sigma(2,3));