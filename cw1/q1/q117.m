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

%% 实验0
[results_origin, time_origin] = method_origin(dwis, bvals, qhat, slice_num);
% 可视化结果m
visualize_results(results_origin, 'Origin Initialization Method');
fprintf('\n===== 方法比较 =====\n');
fprintf('方法0 (随机初始点) 耗时: %.2f 秒\n', time_origin);

%% 实验1
[results_rician, time_rician] = method_rician(dwis, bvals, qhat, slice_num);
% 可视化结果m
visualize_results(results_rician, 'Method with rician noise'); 
fprintf('Method with rician noise耗时: %.2f 秒\n', time_rician);

%% 打印各方法的 RESNORM 平均值
avg_RESNORM_origin = mean(results_origin.RESNORM(~isnan(results_origin.RESNORM)));
avg_RESNORM_rician = mean(results_rician.RESNORM(~isnan(results_rician.RESNORM)));

fprintf('\n===== 方法性能比较 =====\n');
fprintf('方法0 的平均 RESNORM: %.4e\n', avg_RESNORM_origin);
fprintf('方法1 的平均 RESNORM: %.4e\n', avg_RESNORM_rician);