load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]); % 调整维度为[108,145,174,145]
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);

Avox = dwis(:,92,65,72);
% Define a starting point for the non-linear fit
startx = [3.5e+00, 3e-03, 2.5e-01, 0, 0];
% Define various options for the non-linear fitting
% algorithm.
h=optimset('MaxFunEvals',20000,...
 'Algorithm','quasi-newton',...
 'TolX',1e-10,...
 'TolFun',1e-10);
% Now run the fitting
[parameter_hat, RESNORM, EXITFLAG, OUTPUT] = fminunc(@(x) BallStickSSD(x, Avox, bvals, qhat), startx, h);

% 显示拟合参数和 SSD 值
format short e
disp('Fitted parameters (S0, d, f, theta, phi):');
disp(parameter_hat);
disp('RESNORM (SSD):');
disp(RESNORM);

%% 生成拟合图
% 计算模型预测信号
S0 = parameter_hat(1);
d = parameter_hat(2);
f_val = parameter_hat(3);
theta = parameter_hat(4);
phi = parameter_hat(5);

% 计算纤维方向
fibdir = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
fibdotgrad = sum(qhat .* repmat(fibdir, [size(qhat,1), 1]), 2);
S_model = S0 * ( f_val * exp(-bvals*d .* (fibdotgrad.^2)) + (1-f_val) * exp(-bvals * d));

% 这里令 k 为测量编号（1 到 108）
k = (1:length(bvals))';

% 绘图
figure;
hold on;
% 绘制实际测量数据：蓝色点
plot(k, Avox, 'bo', 'MarkerFaceColor','b', 'DisplayName','Data');
% 绘制模型预测数据：红色点
plot(k, S_model, 'ro', 'MarkerFaceColor','r', 'DisplayName','Model');
xlabel('k (Measurement Index)'); ylabel('S (Signal Intensity)');
title(sprintf('Ball-and-Stick Model Fit SSD = %.4e', RESNORM));
legend('Location','best');

% 为图形侧边添加 SSD 值
dim = [0.75 0.6 0.2 0.2]; % 文字框在图内的位置（归一化单位）
str = sprintf('SSD = %.4e', RESNORM);
annotation('textbox', dim, 'String', str, 'FitBoxToText','on', 'BackgroundColor','w');
hold off;