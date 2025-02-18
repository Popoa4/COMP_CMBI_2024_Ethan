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
Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    
%% 构建设计矩阵 Y
Y = build_design_matrix(bvals, qhat);
    
%% 计算DTI参数作为初始点
[S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    
%% 定义自助法参数
num_bootstrap = 1000;     % 自助次数
num_measurements = length(Avox_original); % 108
params_bootstrap = zeros(num_bootstrap, 3); % 存储 [S0, d, f]
    
%% 开始自助法
fprintf('开始自助法（Bootstrap）估计...\n');
parfor i = 1:num_bootstrap % 使用并行计算加速
    % 自助样本生成（有放回抽样）
    indices = randi(num_measurements, num_measurements, 1);
    Avox_boot = Avox_original(indices);
    qhat_boot = qhat(indices, :);
    bvals_boot = bvals(indices);
    Y_boot = build_design_matrix(bvals_boot, qhat_boot);
        
    % 多次拟合以确保全局最小值
    [S0_bs, d_bs, f_bs, ~] = robust_ball_stick_fit(Avox_boot, bvals_boot, qhat_boot, Y_boot, S0_init, d_init, f_init, theta_init, phi_init);
        
    % 存储拟合参数
    params_bootstrap(i, :) = [S0_bs, d_bs, f_bs];
end
    
%% 计算置信区间
ci_2sigma = prctile(params_bootstrap, [2.5 97.5]); % 95%置信区间
ci_95 = prctile(params_bootstrap, [2.5 97.5]);      % 与2-sigma相同
    
% 输出结果
fprintf('参数Bootstrap 95%%置信区间:\n');
fprintf('S0: [%.2f, %.2f]\n', ci_2sigma(1,1), ci_2sigma(2,1));
fprintf('d : [%.6f, %.6f]\n', ci_2sigma(1,2), ci_2sigma(2,2));
fprintf('f : [%.4f, %.4f]\n', ci_2sigma(1,3), ci_2sigma(2,3));
    
    % %% 其他体素的比较（可选）
    % other_voxels = [93, 65, 72; 80, 50, 72; 100, 80, 72];
    % for v = 1:size(other_voxels,1)
    %     voxel = other_voxels(v, :);
    %     Avox = squeeze(dwis(:, voxel(1), voxel(2), voxel(3)));
    %     if all(Avox > 0)
    %         [S0_init_v, d_init_v, f_init_v, theta_init_v, phi_init_v] = dti_initialization_single_voxel(Y, Avox);
    %         params_bs_v = zeros(num_bootstrap, 3);
    %         parfor i = 1:num_bootstrap
    %             indices = randi(num_measurements, num_measurements, 1);
    %             Avox_boot = Avox(indices);
    %             qhat_boot = qhat(indices, :);
    %             bvals_boot = bvals(indices);
    %             Y_boot = build_design_matrix(bvals_boot, qhat_boot);
    %             [S0_bs, d_bs, f_bs, ~] = robust_ball_stick_fit(Avox_boot, bvals_boot, qhat_boot, Y_boot, S0_init_v, d_init_v, f_init_v, theta_init_v, phi_init_v);
    %             params_bs_v(i, :) = [S0_bs, d_bs, f_bs];
    %         end
    %         ci_v = prctile(params_bs_v, [2.5 97.5]);
    %         fprintf('\n体素 [%d, %d, %d] 参数Bootstrap 95%%置信区间:\n', voxel);
    %         fprintf('S0: [%.2f, %.2f]\n', ci_v(1,1), ci_v(2,1));
    %         fprintf('d : [%.6f, %.6f]\n', ci_v(1,2), ci_v(2,2));
    %         fprintf('f : [%.4f, %.4f]\n', ci_v(1,3), ci_v(2,3));
    %     else
    %         fprintf('\n体素 [%d, %d, %d] 信号无效，跳过。\n', voxel);
    %     end
    % end