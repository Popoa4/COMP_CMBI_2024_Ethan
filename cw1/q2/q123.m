%% ====================== 主脚本 (uncertainty_estimation.m) ======================
% 清空环境
clear; clc; close all;

% 加载数据
load('data');         % 包含变量 dwis
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);  % 调整维度为 [108, 145, 174, 145]

load('bvecs');        % bvecs 为 3×108 的矩阵
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);  % 计算b值

% 中间切片
slice_num = 72;

% 选择单个体素
voxel_coords = [92, 65, 72];
Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));

% 构建设计矩阵 Y
Y = build_design_matrix(bvals, qhat);

% 运行不同的不确定性估计方法
% 1. 经典自助法
[ci_bootstrap, time_bootstrap] = bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, 1000);

% 2. 参数自助法
[ci_param_bootstrap, time_param_bootstrap] = parametric_bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, 1000);

% 3. 拉普拉斯方法
[ci_laplace, time_laplace] = laplace_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y);

% 4. MCMC方法
[ci_mcmc, time_mcmc] = mcmc_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y);
% save('bootstrap_ci.mat', 'S0_ci_low', 'S0_ci_high', 'd_ci_low', 'd_ci_high', 'f_ci_low', 'f_ci_high');

% 结果比较
fprintf('\n===== 不同方法的参数置信区间 =====\n');
fprintf('方法\t\tS0置信区间\n');
fprintf('Bootstrap\t[%.2f, %.2f]\n', ci_bootstrap(1,1), ci_bootstrap(2,1));
fprintf('Param Bootstrap\t[%.2f, %.2f]\n', ci_param_bootstrap(1,1), ci_param_bootstrap(2,1));
fprintf('Laplace\t\t[%.2f, %.2f]\n', ci_laplace(1,1), ci_laplace(1,2));
fprintf('MCMC\t\t[%.2f, %.2f]\n', ci_mcmc(1,1), ci_mcmc(2,1));

% 显示各方法的计算时间
fprintf('\n===== 计算时间 =====\n');
fprintf('Bootstrap: %.2f 秒\n', time_bootstrap);
fprintf('Param Bootstrap: %.2f 秒\n', time_param_bootstrap);
fprintf('Laplace: %.2f 秒\n', time_laplace);
fprintf('MCMC: %.2f 秒\n', time_mcmc);
