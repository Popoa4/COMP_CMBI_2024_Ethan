% 清空工作空间和命令行
clear; clc;

%% 数据加载
load('data');         % 包含变量 dwis
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);  % 使数据维度为 : [108 145 174 145]

load('bvecs');        % 假设 bvecs 为 3×108 的矩阵
qhat = bvecs';        % 转置为 108×3，每行一个梯度方向
bvals = 1000 * sum(qhat.*qhat, 2);  % 对应 b 值（此处等效于1000）

%% 设置
voxel_coords = [92, 65, 72]; % 用于测试的体素坐标，之后可以更改
Avox = dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3));

% 初始参数（变换前，同上一题）
startx_original = [3.5e3, 3e-3, 0.25, 0, 0];

% 将初始参数变换到新的参数空间
startx_transformed(1) = sqrt(startx_original(1));
startx_transformed(2) = sqrt(startx_original(2));
startx_transformed(3) = -log((1/startx_original(3)) - 1);
startx_transformed(4) = startx_original(4);
startx_transformed(5) = startx_original(5);

% 设置优化选项(与之前一样，但关闭显示以加快速度)
h = optimset('MaxFunEvals', 20000, ...
             'Algorithm', 'quasi-newton', ...
             'TolX', 1e-10, ...
             'TolFun', 1e-10, ...
             'Display', 'off'); % 关闭显示

%% 多次拟合
num_trials = 150; % 试验次数
RESNORM_values = zeros(1, num_trials);
all_params = zeros(num_trials, 5);

% 定义扰动尺度（基于初始参数的比例）
perturb_scales = [0.2 * startx_transformed(1), ...  % S0 扰动
                   0.2 * startx_transformed(2), ...  % d  扰动
                   0.1* startx_transformed(3), ...                   % f  扰动 (logistic变换后)
                   0.1 * startx_transformed(4), ...              % theta 扰动
                   0.2* startx_transformed(5)];                % phi   扰动


for i = 1:num_trials
    % 生成随机扰动 (正态分布)
    perturbation = randn(1, 5) .* perturb_scales;
    
    % 新的起点
    current_startx = startx_transformed + perturbation;

    % 对theta和phi取模，确保在有效范围
    current_startx(4) = mod(current_startx(4), pi);        % theta ∈ [0, pi)
    current_startx(5) = mod(current_startx(5), 2*pi);      % phi ∈ [0, 2pi)
    % 进行拟合
    [param_hat_trans, RESNORM_trans, ~, ~] = fminunc(@(x) BallStickSSD_transformed(x, Avox, bvals, qhat), current_startx, h);
    % [parameter_hat_trans, RESNORM_trans, EXITFLAG_trans, OUTPUT_trans] = fminunc(@(x) BallStickSSD_transformed(x, Avox, bvals, qhat), startx_transformed, h);
    % disp(i);
    % disp(RESNORM_trans);
    % 记录结果
    RESNORM_values(i) = RESNORM_trans;
    all_params(i, :) = param_hat_trans; % 记录变换后参数值
end

%% 分析结果
% disp(RESNORM_values);
[min_RESNORM, min_index] = min(RESNORM_values);
best_params_trans = all_params(min_index, :);

% 将最佳参数反变换回原始空间
S0_best    = best_params_trans(1)^2;
d_best     = best_params_trans(2)^2;
f_best     = 1 / (1 + exp(-best_params_trans(3)));
theta_best = best_params_trans(4);
phi_best   = best_params_trans(5);

best_params_original = [S0_best, d_best, f_best, theta_best, phi_best];
plot_fit(phi_best, theta_best, S0_best, f_best, d_best, bvals, qhat, Avox, min_RESNORM);

fprintf('最小 RESNORM 值: %.4e\n', min_RESNORM);
fprintf('对应的原始空间参数 (S0, d, f, theta, phi):\n');
disp(best_params_original);

% 找到最小RESNORM的试验比例
tolerance = 1e-4;
% min_local = min(local_RESNORM);
% success_rate = sum(abs(local_RESNORM - min_local) <= tolerance) / num_trials;
% is_minimal = abs(RESNORM_values - min_RESNORM) <= tolerance * max(1, min_RESNORM);

proportion_best = sum(abs(RESNORM_values - min_RESNORM) <= tolerance) / num_trials;
disp(sum(abs(RESNORM_values - min_RESNORM) < tolerance));
fprintf('找到接近最小 RESNORM 值的试验比例: %.2f\n', proportion_best);


% 估计达到95%置信度所需的试验次数
% p = proportion_best
% 1 - (1-p)^n >= 0.95  =>  n >= log(0.05) / log(1-p)
if proportion_best > 0 && proportion_best < 1  % 避免除零或对数错误
    n_95 = ceil(log(0.05) / log(1 - proportion_best));
    fprintf('达到95%%置信度所需的试验次数: %d\n', n_95);
else
    fprintf('无法估计达到95%%置信度所需的试验次数 (比例为0或1).\n');
end

%% 多体素验证（新增）
test_voxels = [93,65,72;   % 相邻体素
               80,50,70;   % 不同位置
               100,80,72]; % 边缘体素

for v = 1:size(test_voxels,1)
    current_voxel = test_voxels(v,:);
    Avox = dwis(:,current_voxel(1),current_voxel(2),current_voxel(3));
    
    % 快速验证（减少试验次数）
    num_trials = 100;
    local_RESNORM = zeros(1,num_trials);
    
    parfor i = 1:num_trials
        perturbation = randn(1,5).*perturb_scales;
        [~, RESNORM] = fminunc(@(x)BallStickSSD_transformed(x,Avox,bvals,qhat),...
                              startx_transformed + perturbation, h);
        local_RESNORM(i) = RESNORM;
    end
    
    min_local = min(local_RESNORM);
    success_rate = sum(abs(local_RESNORM - min_local) <= tolerance) / num_trials;
    
    fprintf('\n体素 [%d,%d,%d]:\n', current_voxel);
    fprintf('最小RESNORM: %.3e 成功比例: %.2f\n', min_local, success_rate);
end
