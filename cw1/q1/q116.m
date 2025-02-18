% 比较三种方法
% 1. DTI初始点 + 随机扰动
% 2. fmincon约束优化
% 3. 解析导数

% metrics
% 1. 单个体素平均计算时间
% 2. 全局最小值找到的概率
% 3. 切片映射总时间
%% 数据加载
% 初始化环境和数据加载
clear; clc; close all;
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]); % 调整维度为[108,145,174,145]
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);
slice_num = 72;
%% 实验1
[results_dti, time_dti] = method_dti_initial(dwis, bvals, qhat, slice_num);
% 可视化结果m
visualize_results(results_dti, 'DTI Initialization Method');
fprintf('\n===== 方法比较 =====\n');
fprintf('方法1 (DTI初始点) 耗时: %.2f 秒\n', time_dti);

%% 实验2
[results_fmincon, time_fmincon] = method_fmincon(dwis, bvals, qhat, slice_num);
visualize_results(results_fmincon, 'fmincon Constrained Method');
fprintf('方法2 (fmincon约束) 耗时: %.2f 秒\n', time_fmincon);

%% 实验3
[results_analytic, time_analytic] = method_analytic_grad(dwis, bvals, qhat, slice_num);
visualize_results(results_analytic, 'Analytic Gradient Method');
fprintf('方法3 (解析梯度) 耗时: %.2f 秒\n', time_analytic);

%% 打印各方法的 RESNORM 平均值
avg_RESNORM_dti = mean(results_dti.RESNORM(~isnan(results_dti.RESNORM)));
avg_RESNORM_fmincon = mean(results_fmincon.RESNORM(~isnan(results_fmincon.RESNORM)));
avg_RESNORM_analytic = mean(results_analytic.RESNORM(~isnan(results_analytic.RESNORM)));
fprintf('\n===== 方法性能比较 =====\n');
fprintf('方法1 的平均 RESNORM: %.4e\n', avg_RESNORM_dti);
fprintf('方法2 的平均 RESNORM: %.4e\n', avg_RESNORM_fmincon);
fprintf('方法3 的平均 RESNORM: %.4e\n', avg_RESNORM_analytic);