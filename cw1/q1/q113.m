% 清空工作空间和命令行
clear; clc;

%% 数据加载
load('data');         % 包含变量 dwis
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);  % 使数据维度为 : [108 145 174 145]

load('bvecs');        % 假设 bvecs 为 3×108 的矩阵
qhat = bvecs';        % 转置为 108×3，每行一个梯度方向
bvals = 1000 * sum(qhat.*qhat, 2);  % 对应 b 值（此处等效于1000）

%% 选择单个体素（与 lec 一致，体素索引为 (92,65,72)）
Avox = dwis(:,92,65,72);  % 108个测量值
% 注意：确保 Avox 中信号均大于零（否则取对数或进行其它计算时会有问题）

% 初始参数（变换前）
startx_original = [3.5e3, 3e-3, 0.25, 0, 0];

% 将初始参数变换到新的参数空间
% startx_transformed(1) = sqrt(startx_original(1));
% startx_transformed(2) = sqrt(startx_original(2));
% startx_transformed(3) = -log((1/startx_original(3)) - 1);

startx_transformed(1) = sqrt(startx_original(1));
startx_transformed(2) = sqrt(startx_original(2));
startx_transformed(3) = -log((1/startx_original(3)) - 1);
startx_transformed(4) = startx_original(4);
startx_transformed(5) = startx_original(5);


%% 设置优化选项
h = optimset('MaxFunEvals', 20000, ...
             'Algorithm', 'quasi-newton', ...
             'TolX', 1e-10, ...
             'TolFun', 1e-10, ...
             'Display', 'iter');

% 使用变换后的目标函数进行拟合
[parameter_hat_trans, RESNORM_trans, EXITFLAG_trans, OUTPUT_trans] = fminunc(@(x) BallStickSSD_transformed(x, Avox, bvals, qhat), startx_transformed, h);

% 将拟合结果变换回原始参数空间
S0_fit    = parameter_hat_trans(1)^2;
d_fit     = parameter_hat_trans(2)^2;
f_fit     = 1 / (1 + exp(-parameter_hat_trans(3)));
%     startx_transformed(1) = sqrt(startx_original(1));
% startx_transformed(2) = sqrt(startx_original(2));
% startx_transformed(3) = -log((1/startx_original(3)) - 1);
theta_fit = parameter_hat_trans(4);
phi_fit   = parameter_hat_trans(5);

parameter_hat_original = [S0_fit, d_fit, f_fit, theta_fit, phi_fit];

% 显示变换后的拟合参数和SSD
% disp('Fitted parameters (transformed, then back to original):');
disp('Fitted parameters (S0, d, f, theta, phi):');
disp(parameter_hat_original);
disp('RESNORM (transformed):');
disp(RESNORM_trans);

%% 生成拟合图（使用变换后的参数）
plot_fit(phi_fit, theta_fit, S0_fit, f_fit, d_fit, bvals, qhat, Avox, RESNORM_trans)
% 计算模型信号
% fibdir = [cos(phi_fit)*sin(theta_fit), sin(phi_fit)*sin(theta_fit), cos(theta_fit)];
% fibdotgrad = sum(qhat .* repmat(fibdir, [size(qhat,1), 1]), 2);
% S_model = S0_fit * ( f_fit * exp(-bvals*d_fit .* (fibdotgrad.^2)) + (1-f_fit) * exp(-bvals*d_fit) );
% % 这里令 k 为测量编号（1 到 108）
% k = (1:length(bvals))';
% 
% % 绘图
% figure;
% hold on;
% % 绘制实际测量数据：蓝色点
% plot(k, Avox, 'bo', 'MarkerFaceColor','b', 'DisplayName','Data');
% % 绘制模型预测数据：红色点
% plot(k, S_model, 'ro', 'MarkerFaceColor','r', 'DisplayName','Model');
% xlabel('k (Measurement Index)');
% ylabel('S (Signal Intensity)');
% title(sprintf('Ball-and-Stick Model Fit (Transformed)   SSD = %.4e', RESNORM_trans)); % 注意这里的SSD也应为变换后的
% legend('Location','best');
% 
% % 为图形侧边添加 SSD 值
% dim = [0.75 0.6 0.2 0.2];
% str = sprintf('SSD = %.4e', RESNORM_trans);
% annotation('textbox', dim, 'String', str, 'FitBoxToText','on', 'BackgroundColor','w');
% hold off;

