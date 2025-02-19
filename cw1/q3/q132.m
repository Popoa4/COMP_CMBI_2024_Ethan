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
[params_bs, resnorm_bs] = fit_ball_stick(meas, bvals, qhat);
[params_zs, resnorm_zs] = fit_zeppelin_stick(meas, bvals, qhat);
[params_zt, resnorm_zt] = fit_tortuosity(meas, bvals, qhat);

% 结果展示
fprintf('===== 模型比较结果 =====\n');
fprintf('Ball and Stick RESNORM: %.4f\n', resnorm_bs);
fprintf('Zeppelin and Stick RESNORM: %.4f\n', resnorm_zs);
fprintf('Tortuosity Model RESNORM: %.4f\n', resnorm_zt);

%% 可视化拟合效果
figure;
subplot(3,1,1);
plot_model_fit(params_bs, 'Ball and Stick', meas, bvals, qhat);
subplot(3,1,2);
plot_model_fit(params_zs, 'Zeppelin and Stick', meas, bvals, qhat);
subplot(3,1,3);
plot_model_fit(params_zt, 'Tortuosity Model', meas, bvals, qhat);

function plot_model_fit(params, model_name, meas, bvals, grad_dirs)
    best_S0 = params(1);
    best_d = params(2);
    best_f = params(3);
    best_theta = params(4);
    best_phi = params(5);
    
    % 生成模型预测值
    fibdir = [cos(best_phi)*sin(best_theta), sin(best_phi)*sin(best_theta), cos(best_theta)];
    fibdotgrad = sum(grad_dirs .* fibdir, 2);
    S_pred = best_S0 * (best_f * exp(-bvals*best_d .* fibdotgrad.^2) + (1-best_f)*exp(-bvals*best_d));
    
    % 绘制预测 vs 实际信号
    % figure;
    scatter(bvals, meas, 5, 'b', 'filled'); hold on;
    scatter(bvals, S_pred, 5, 'r', 'filled');
    xlabel('b-value (s/mm²)');
    ylabel('Signal');
    legend('Measured', 'Predicted');
    title(model_name);
end