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
voxel_coords = [92, 65, 72];
Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    
%% 构建设计矩阵 Y (复用Q1.1.1)
Y = build_design_matrix(bvals, qhat);
    
%% DTI初始化获取起始点
[S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    
%% 定义MCMC参数
num_iterations = 10000;    % 迭代次数
burn_in = 2000;            % 烧入期
thinning = 10;             % 薄化间隔
    
% 各参数的先验分布边界
lb = [0, 0, 0, 0, 0];      % S0, d, f, theta, phi 下限
ub = [Inf, Inf, 1, pi, 2*pi]; % 上限
    
% 参数向量 [S0, d, f, theta, phi]
current_params = [S0_init, d_init, f_init, theta_init, phi_init];
    
% 先验分布（假设均匀分布）
log_prior = @(x) (x(1) > 0 && x(2) > 0 && x(3) >=0 && x(3) <=1 && ...
                  x(4) >=0 && x(4) <=pi && x(5) >=0 && x(5) <=2*pi) * 0 + ...
                  (-Inf) * (~(x(1) > 0 && x(2) > 0 && x(3) >=0 && x(3) <=1 && ...
                              x(4) >=0 && x(4) <=pi && x(5) >=0 && x(5) <=2*pi));
                  
% 似然函数：假设噪声为高斯分布，标准差已知（例如，200）
sigma_noise = 200;
log_likelihood = @(x) -0.5 * sum((Avox_original - BallStick_model(x, bvals, qhat)).^2) / sigma_noise^2;
    
% 后验分布（对数）
log_posterior = @(x) log_prior(x) + log_likelihood(x);
    
%% 定义提议分布（高斯扰动）
proposal_std = [100, 1e-4, 0.05, 0.1, 0.1]; % 调整步长
proposal = @(x) x + proposal_std .* randn(1,5);
    
%% 初始化MCMC采样
samples = zeros(num_iterations, 5);
acceptance = 0;
    
%% MCMC采样循环
fprintf('开始MCMC采样...\n');
for i = 1:num_iterations
    % 提议新的参数
    proposed_params = proposal(current_params);
        
    % 计算后验概率
    log_posterior_current = log_posterior(current_params);
    log_posterior_proposed = log_posterior(proposed_params);
        
    % 计算接受率
    log_accept_ratio = log_posterior_proposed - log_posterior_current;
    if log(rand) < log_accept_ratio
        % 接受提议
        current_params = proposed_params;
        acceptance = acceptance + 1;
    end
        
    % 记录样本
    samples(i, :) = current_params;
end
    
%% 后处理
% 计算接受率
acceptance_rate = acceptance / num_iterations;
fprintf('MCMC接受率: %.2f%%\n', acceptance_rate * 100);
    
% 丢弃烧入期和进行薄化
samples_post = samples(burn_in:thinning:end, :);
    
%% 计算置信区间
ci_95_MCMC = prctile(samples_post, [2.5 97.5]);
    
fprintf('参数MCMC 95%%置信区间:\n');
fprintf('S0: [%.2f, %.2f]\n', ci_95_MCMC(1,1), ci_95_MCMC(2,1));
fprintf('d : [%.6f, %.6f]\n', ci_95_MCMC(1,2), ci_95_MCMC(2,2));
fprintf('f : [%.4f, %.4f]\n', ci_95_MCMC(1,3), ci_95_MCMC(2,3));
    
%% 可视化采样结果
figure;
subplot(3,1,1);
histogram(samples_post(:,1), 50); title('S0 Posterior Distribution');
subplot(3,1,2);
histogram(samples_post(:,2), 50); title('d Posterior Distribution');
subplot(3,1,3);
histogram(samples_post(:,3), 50); title('f Posterior Distribution');
    