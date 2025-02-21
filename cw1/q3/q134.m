% 模型比较主程序
% % 加载数据
% [meas, bvals, grad_dirs] = load_isbi_data();
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

% 拟合各模型
[params_b2s, resnorm_b2s] = fit_ball_two_sticks(meas, bvals, qhat);
[params_b3s, resnorm_b3s] = fit_ball_multi_sticks(meas, bvals, qhat, 3);
[params_b5s, resnorm_b5s] = fit_ball_multi_sticks(meas, bvals, qhat, 5);
[params_b10s, resnorm_b10s] = fit_ball_multi_sticks(meas, bvals, qhat, 10);


% 结果展示
fprintf('===== 模型比较结果 =====\n');
fprintf('Ball and Two Stick RESNORM: %.4f\n', resnorm_b2s);
fprintf('Ball and Three Stick RESNORM: %.4f\n', resnorm_b3s);
fprintf('Ball and Five Stick RESNORM: %.4f\n', resnorm_b5s);
fprintf('Ball and Ten Stick RESNORM: %.4f\n', resnorm_b10s);
