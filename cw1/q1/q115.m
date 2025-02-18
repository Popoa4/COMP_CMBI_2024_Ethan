%% 数据加载（如果尚未加载）
clear;clc;
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);

load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat.*qhat, 2);

%% 参数设置
slice_num = 72;
[x_dim, y_dim] = size(dwis, 2:3);

% 初始参数 (变换前)
startx_original = [3.5e3, 3e-3, 0.25, 0, 0];
startx_transformed(1) = sqrt(startx_original(1));
startx_transformed(2) = sqrt(startx_original(2));
startx_transformed(3) = -log((1/startx_original(3)) - 1);
startx_transformed(4) = startx_original(4);
startx_transformed(5) = startx_original(5);

% 优化选项
h = optimset('MaxFunEvals', 20000, ...
             'Algorithm', 'quasi-newton', ...
             'TolX', 1e-10, ...
             'TolFun', 1e-10, ...
             'Display', 'off'); % 关闭显示
    

% 多次试验参数
num_trials = 3; % 根据Q1.1.4结果调整
perturb_scales = [0.2 * startx_transformed(1), ...
                  0.2 * startx_transformed(2), ...
                  0.2, 0.1 * pi, 0.2*pi];
tolerance = 1e-6; % 容差
%% 参数映射图初始化
S0_map    = zeros(x_dim, y_dim);
d_map     = zeros(x_dim, y_dim);
f_map     = zeros(x_dim, y_dim);
RESNORM_map = zeros(x_dim, y_dim);
theta_map = zeros(x_dim, y_dim); % 用于纤维方向
phi_map   = zeros(x_dim, y_dim); % 用于纤维方向
%% 遍历切片
for x = 1:x_dim
    for y = 1:y_dim
        voxel = squeeze(dwis(:, x, y, slice_num));

        % 检查信号有效性
        if all(voxel > 0)
            Avox = voxel;
            RESNORM_values = zeros(1, num_trials);
            all_params = zeros(num_trials, 5);
            
            % 多次拟合
            for i = 1:num_trials
                % disp(i);
                perturbation = randn(1, 5) .* perturb_scales;
                current_startx = startx_transformed + perturbation;
                % 对theta和phi取模，确保在有效范围
                current_startx(4) = mod(current_startx(4), pi);        % theta ∈ [0, pi)
                current_startx(5) = mod(current_startx(5), 2*pi);      % phi ∈ [0, 2pi)
                try
                    [param_hat_trans, RESNORM_trans, ~, ~] = fminunc(@(x) BallStickSSD_transformed(x, Avox, bvals, qhat), current_startx, h);
    
                    RESNORM_values(i) = RESNORM_trans;
                    all_params(i, :) = param_hat_trans;
                catch ME
                    % 如果fminunc出错, 则:
                    fprintf('Trial %d 出现错误: %s\n', i, ME.message);
                    
                    % 1. 将RESNORM设置为一个很大的值, 以便在后续比较中不会被选为最小值
                    RESNORM_values(i) = Inf;
                    
                    % 2.  (可选) 你也可以选择将参数设置为某个默认值, 例如初始值
                    all_params(i, :) = startx_transformed; 
                    
                    % 3.  (可选) 如果你想完全跳过这次迭代, 可以使用 'continue'
                    % continue; 
                end
            end

            % 找出最佳拟合
            is_minimal = abs(RESNORM_values - min(RESNORM_values)) <= tolerance * max(1, min(RESNORM_values));
            best_params_trans = all_params(find(is_minimal, 1), :);

            % 反变换
            S0_best    = best_params_trans(1)^2;
            d_best     = best_params_trans(2)^2;
            f_best     = 1 / (1 + exp(-best_params_trans(3)));
            theta_best = best_params_trans(4);
            phi_best   = best_params_trans(5);

            % 存储结果
            S0_map(x, y)    = S0_best;
            d_map(x, y)     = d_best;
            f_map(x, y)     = f_best;
            RESNORM_map(x, y) = min(RESNORM_values);
            theta_map(x,y) = theta_best;
            phi_map(x,y) = phi_best;
       else
            % 对于无效信号, 设置为0或NaN (根据需要)
            S0_map(x, y)    = 0; % 或 NaN
            d_map(x, y)     = 0; % 或 NaN
            f_map(x, y)     = 0; % 或 NaN
            RESNORM_map(x, y) = NaN;
            theta_map(x,y) = NaN;
            phi_map(x,y) = NaN;

        end
    end
end

%% 显示参数映射图
figure;
subplot(2,2,1); imagesc(flipud(S0_map')); axis image off; colormap gray; title('S0 Map'); colorbar;
subplot(2,2,2); imagesc(flipud(d_map')); axis image off; colormap gray; title('d Map'); colorbar;
subplot(2,2,3); imagesc(flipud(f_map')); axis image off; colormap gray; title('f Map'); colorbar;
subplot(2,2,4); imagesc(flipud(RESNORM_map')); axis image off; colormap jet; title('RESNORM Map'); colorbar;

%% 纤维方向图

% 计算纤维方向 (并用f加权)
[X, Y] = meshgrid(1:y_dim, 1:x_dim); % 注意这里调换了x_dim和y_dim
fibdir_x = f_map .* sin(theta_map) .* cos(phi_map);
fibdir_y = f_map .* sin(theta_map) .* sin(phi_map);

% 绘制纤维方向 (quiver)
figure;
quiver(X, Y, flipud(fibdir_x), flipud(fibdir_y), 2.5);
% 参数2.5控制箭头长度的缩放, 可根据图像效果调整
title('Fiber Directions (weighted by f)');
axis image;
