%% ====================== 参数自助法函数 (parametric_bootstrap_uncertainty.m) ======================
function [ci_param_bootstrap, total_time] = parametric_bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, num_bootstrap)
    tic;
    
    % 原始信号
    Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    
    % 使用DTI参数作为初始点
    [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    
    % 拟合原始数据，获取参数估计
    [S0_est, d_est, f_est, ~] = robust_ball_stick_fit(Avox_original, bvals, qhat, Y, S0_init, d_init, f_init, theta_init, phi_init);
    % 重新获取最优参数的变换表示
    % S0_est = sqrt(S0_est);
    % d_est = sqrt(d_est);
    % f_est = -log((1/f_est) - 1);
    % theta_trans = theta_init;
    % phi_trans = phi_init;
    % params_trans = [S0_trans, d_trans, f_trans, theta_trans, phi_trans];

    % 生成样本
    params_bootstrap = zeros(num_bootstrap, 3);
    
    % Precompute fiber directions
    % fibdir = [cos(phi_init)*sin(theta_init), sin(phi_init)*sin(theta_init), cos(theta_init)];
    % fibdotgrad = sum(qhat .* repmat(fibdir, [size(qhat,1), 1]), 2);
    fibdir = [cos(phi_init)*sin(theta_init), sin(phi_init)*sin(theta_init), cos(theta_init)];
    fibdotgrad = sum(qhat .* fibdir, 2);
    fprintf('开始参数自助法（Parametric Bootstrap）估计，共 %d 次...\n', num_bootstrap);
    
    parfor i = 1:num_bootstrap
        % 生成合成数据
        % noise = randn(length(bvals), 1) * 200; % 假设噪声标准差为200
        % S_model = S0_est * (f_est * exp(-bvals * d_est .* (fibdotgrad.^2)) + (1 - f_est) * exp(-bvals * d_est));
        % Avox_boot = S_model + noise;
        % Avox_boot(Avox_boot < 1e-10) = 1e-10; % 避免log(0)
        % a) 生成合成数据
        S_model = S0_est * ( f_est * exp(-bvals .* d_est .* (fibdotgrad.^2)) ...
                           + (1 - f_est) * exp(-bvals .* d_est) );
        noise    = randn(size(S_model)) * 200;
        Avox_boot= S_model + noise;
        Avox_boot(Avox_boot < 1e-10) = 1e-10;
        % 拟合
        [S0_bs, d_bs, f_bs, ~] = robust_ball_stick_fit(Avox_boot, bvals, qhat, Y, S0_init, d_init, f_init, theta_init, phi_init);

        % S0_bs = sqrt(S0_bs);
        % d_bs = sqrt(d_bs);
        % f_bs = -log((1/f_bs) - 1);
        % 存储参数
        params_bootstrap(i, :) = [S0_bs, d_bs, f_bs];
    end
    
    % 计算置信区间（2-sigma ~ 95%）
    ci_param_bootstrap = prctile(params_bootstrap, [2.5 97.5]);
    
    total_time = toc;
end
