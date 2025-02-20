clear; clc; close all;
%% 1. 加载数据和协议
% 加载扩散信号数据（注意文件名，与下载的文件一致）
fid = fopen('isbi2015_data_normalised.txt', 'r', 'b');
fgetl(fid); % 跳过头部
D = fscanf(fid, '%f', [6, inf])'; % 数据尺寸为 [N_measurements × 6]
fclose(fid);

% 例如，这个数据文件中包含6个体素的测量，选择第一个体素的测量
meas = D(:,1); % meas 为 3612×1 信号向量

% 加载协议文件
fid = fopen('isbi2015_protocol.txt', 'r', 'b');
fgetl(fid); % 跳过头部
A = fscanf(fid, '%f', [7, inf]);
fclose(fid);

% 创建协议变量
grad_dirs = A(1:3,:); % 3×N_measurements ，每列代表一个测量的梯度方向
qhat = grad_dirs';
G = A(4,:)'; % 梯度幅值
delta = A(5,:)'; % 梯度持续时间
smalldel = A(6,:)'; % 梯度上升时间
TE = A(7,:)'; % 回波时间（可能用不上）
GAMMA = 2.675987E8; % Gyromagnetic ratio, in rad/T/s

% 计算 b 值 (单位为 s/m^2), 然后转换为 s/mm^2
bvals = ((GAMMA * smalldel .* G).^2).*(delta - smalldel/3);
bvals = bvals/1e6; % 现在 bvals 单位为 s/mm^2

%% 模型拟合设置
% 针对新数据，测量数目为 N = 3612。噪声标准差为 sigma = 0.04 (signal 无量纲)
sigma_noise = 0.04;
N = length(meas);

% 我们使用与 Q1.1.3 类似的参数变换方法将参数编码为：
% S0 = x(1)^2, d = x(2)^2, f = 1/(1+exp(-x(3))), theta = x(4), phi = x(5)
% 与以前不同，这里我们需要考虑数据噪声很小（sigma=0.04）和测量数目大，因此
% 整体残差（RESNORM）预计大约为： N * sigma_noise^2 ≈ 3612 *0.0016 ≈ 5.78

% 为此我们需要找到合适的初始点。由于数据较多，我们建议先用线性扩散张量模型拟合一遍。
% 这里我们直接采用一组大致合理的起始参数：
% S0_guess ~ mean(meas) (注意：数据已归一化，所以 S0 大概在一个合适的量级)
% d_guess ~ 0.001 – 0.005 (单位： mm^2/s)
% f_guess ~ 0.3 (体积分数)
% theta,phi 根据梯度方向偏好可初值设定为 pi/4
%% 基于DTI的初始化
% 构建设计矩阵
Y = [ones(length(bvals),1), -bvals.*(grad_dirs(1,:)'.^2), -2*bvals.*grad_dirs(1,:)'.*grad_dirs(2,:)', ...
     -2*bvals.*grad_dirs(1,:)'.*grad_dirs(3,:)', -bvals.*(grad_dirs(2,:)'.^2), ...
     -2*bvals.*grad_dirs(2,:)'.*grad_dirs(3,:)', -bvals.*(grad_dirs(3,:)'.^2)];

% 线性回归求解DTI参数
x_dti = Y \ log(meas);
S0_init = exp(x_dti(1));
D = [x_dti(2) x_dti(3) x_dti(4); 
     x_dti(3) x_dti(5) x_dti(6); 
     x_dti(4) x_dti(6) x_dti(7)];

% 提取主方向
[V, L] = eig(D);
[~, idx] = max(diag(L));
main_dir = V(:,idx);
theta_init = acos(main_dir(3));
phi_init = atan2(main_dir(2), main_dir(1));

% 初始参数设置
startx_original = [S0_init, trace(D)/3, 0.5, theta_init, phi_init]; % d=MD, f=0.5
% startx_transformed = [sqrt(startx_original(1)), sqrt(startx_original(2)), ...
%                       log(startx_original(3)/(1-startx_original(3))), ...
%                       theta_init, phi_init];

% S0_guess = mean(meas(bvals < 50)); % 例如，若均值约为 1
% % return;
% d_guess = 2e-3; % 合理范围
% f_guess = 0.3;
% theta_guess = pi/4;
% phi_guess = pi/4;
% startx_original = [S0_guess, d_guess, f_guess, theta_guess, phi_guess];

%% 定义优化选项（注意新数据的噪声水平与测量数目）
options = optimoptions('fmincon',...
    'Algorithm', 'sqp',...       % 序列二次规划算法
    'MaxFunctionEvaluations', 1e4,...
    'StepTolerance', 1e-8,...
    'OptimalityTolerance', 1e-6,...
    'Display', 'iter');
%% 约束条件设置
% 参数边界约束 [S0, d, f, theta, phi]
lb = [0,   0,   0,   0,    0];   % 下限
ub = [Inf, Inf, 1,   pi, 2*pi];  % 上限

%% 可视化残差空间分布
% 计算残差
% S_pred = BallStickSSD_Enhanced(startx_transformed, meas, bvals, qhat);
% residuals = meas - S_pred;
% 
% % 按梯度方向分组显示残差
% [~,~,bin] = histcounts(atan2(qhat(:,2), qhat(:,1)), linspace(-pi,pi,9));
% % disp(bin);
% figure;
% boxplot(residuals, bin);
% xlabel('梯度方向组'); ylabel('残差');
% title('不同梯度方向的残差分布');
% return;
%% 多次拟合尝试，收集 RESNORM
num_trials = 100; % 为提高全局最小的识别率可适当增加尝试次数
RESNORM_values = zeros(num_trials,1);
params_all = zeros(num_trials,5);
% perturb_scales = [5*startx_transformed(1), 5*startx_transformed(2), 5*startx_transformed(3), 5*startx_transformed(4), 5*startx_transformed(5)]; % 根据新数据调整扰动幅度
perturb_scales = [0.5 * startx_original(1), ...  % S0 扰动
                   0.5 * startx_original(2), ...  % d  扰动
                   0.5* startx_original(3), ...                   % f  扰动 (logistic变换后)
                   0.5 * startx_original(4), ...              % theta 扰动
                   0.5* startx_original(5)];                % phi   扰动
for i = 1:num_trials
    perturbation = randn(1,5).*perturb_scales;
    current_startx = startx_original + perturbation;
    try
        % [params_hat_trans, RESNORM, ~, ~] = fminunc(@(x) BallStickSSD_transformed(x, meas, bvals, qhat), ...
        % current_startx, h);
        [params_hat_trans, RESNORM] = fmincon(@(x)BallStickSSD_Enhanced(x,meas,bvals,qhat),...
                          current_startx, [], [], [], [], lb, ub, @nonlcon, options);
        
    catch ME
        fprintf('Trial %d error: %s\n', i, ME.message);
        RESNORM = Inf;
        params_hat_trans = current_startx;
    end
    % fprintf('RESNORM = %.4e\n', RESNORM);
    RESNORM_values(i) = RESNORM;
    % S0_fit    = params_hat_trans(1)^2;
    % d_fit     = params_hat_trans(2)^2;
    % f_fit     = 1 / (1 + exp(-params_hat_trans(3)));
    % theta_fit = params_hat_trans(4);
    % phi_fit   = params_hat_trans(5);
    params_all(i,:) = params_hat_trans;
    % disp(params_all(i,:));
end

% 定义一个相对容差（如差值小于1e-6）判断是否找到同一解：
tol_for_global = 1e-8;
min_RESNORM = min(RESNORM_values);
global_trials = sum(abs(RESNORM_values - min_RESNORM) <= tol_for_global * max(1, min_RESNORM));
fprintf('最小 RESNORM = %.4e\n', min_RESNORM);
fprintf('在 %d 次试验中有 %d 次找到相同的全局最小值 (比例 = %.2f%%)\n', num_trials, global_trials, 100*global_trials/num_trials);

%% 获取最佳拟合参数并反变换回原始参数
% best_idx = find(abs(RESNORM_values - min_RESNORM) <= tol_for_global*max(1,min_RESNORM), 1);
% disp(params_all(best_idx,:));
% [S0_fit, d_fit, f_fit] = params_all(best_idx,:);

[min_RESNORM, min_index] = min(RESNORM_values);
best_params_trans = params_all(min_index, :);
disp(best_params_trans);
% [S0_fit, d_fit, f_fit] = transform_params(best_params_trans);
% [S0_fit, d_fit, f_fit] = best_params_trans;
% 输出结果
fprintf('最佳拟合参数（原始空间）:\n');
fprintf('S0 = %.2f\n', best_params_trans(1));
fprintf('d = %.6f\n', best_params_trans(2));
fprintf('f = %.4f\n', best_params_trans(3));
fprintf('\n注：预计 RESNORM ≈ %f （36120.04^2 ≈ 5.78），本次最小 RESNORM = %.4e\n', N*sigma_noise^2, min_RESNORM);



%% 可视化拟合效果
best_S0 = best_params_trans(1);
best_d = best_params_trans(2);
best_f = best_params_trans(3);
best_theta = best_params_trans(4);
best_phi = best_params_trans(5);

% 生成模型预测值
fibdir = [cos(best_phi)*sin(best_theta), sin(best_phi)*sin(best_theta), cos(best_theta)];
fibdotgrad = sum(grad_dirs' .* fibdir, 2);
S_pred = best_S0 * (best_f * exp(-bvals*best_d .* fibdotgrad.^2) + (1-best_f)*exp(-bvals*best_d));

% 绘制预测 vs 实际信号
figure;
scatter(bvals, meas, 5, 'b', 'filled'); hold on;
scatter(bvals, S_pred, 5, 'r', 'filled');
xlabel('b-value (s/mm²)');
ylabel('Signal');
legend('Measured', 'Predicted');
title('Ball-and-Stick Model Fit Quality');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c, ceq] = nonlcon(x)
    % 非线性不等式约束 c(x) ≤ 0
    c = []; 
    % 非线性等式约束 ceq(x) = 0
    ceq = [];
    % 示例：强制纤维方向与Z轴夹角小于60度
    % theta = x(4);
    % c = [theta - pi/3]; % theta ≤ π/3
end