%% ====================== 拉普拉斯方法函数 (laplace_uncertainty.m) ======================
function [ci_laplace, total_time] = laplace_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y)
    tic;
    
    % 原始信号
    Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    
    % 使用DTI参数作为初始点
    [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    
    % 拟合原始数据，获取参数估计
    [S0_est, d_est, f_est, ~] = robust_ball_stick_fit(Avox_original, bvals, qhat, Y, S0_init, d_init, f_init, theta_init, phi_init);
    
    % 重新获取最优参数的变换表示
    S0_trans = sqrt(S0_est);
    d_trans = sqrt(d_est);
    f_trans = -log((1/f_est) - 1);
    theta_trans = theta_init;
    phi_trans = phi_init;
    params_trans = [S0_trans, d_trans, f_trans, theta_trans, phi_trans];

    % % 定义带梯度的目标函数
    % function [f, g] = BallStickSSD_with_grad(x)
    %     [f, g] = BallStickSSD_transformed_with_grad(x, Avox_original, bvals, qhat);
    % end
    
    
    % 设置优化选项: 告知fminunc提供梯度
    % options = optimset('Display', 'off', 'Algorithm', 'quasi-newton', 'SpecifyObjectiveGradient', true);
    options = optimoptions('fminunc', ...
        'Display', 'off', ...
        'Algorithm', 'quasi-newton', ...
        'SpecifyObjectiveGradient', true);
    
    % 执行拟合并获取Hessian矩阵
    % [params_hat_trans, ~, ~, ~, ~, Hessian] = fminunc(@BallStickSSD_grad, params_trans, options);
    [params_hat_trans, ~, ~, ~, ~, Hessian] = fminunc(@(x) BallStickSSD_grad(x, Avox_original, bvals, qhat), params_trans, options);
    
    
    
    % 计算协方差矩阵（Hessian的逆）
    cov_matrix = inv(Hessian);
    
    % 置信区间（95%）
    std_devs = sqrt(diag(cov_matrix(1:3, 1:3))); % 仅提取 S0, d, f 的方差
    
    ci_laplace = zeros(3,2);
    % ci_laplace(1, :) = params_hat_trans(1:3) - 1.96 * sqrt(diag(cov_matrix(1:3, 1:3)));
    % ci_laplace(2, :) = params_hat_trans(1:3) + 1.96 * sqrt(diag(cov_matrix(1:3, 1:3)));
    ci_laplace(:, 1) = params_hat_trans(1:3)' - 1.96 * std_devs; % 下限
    ci_laplace(:, 2) = params_hat_trans(1:3)' + 1.96 * std_devs; % 上限
    
    % 反变换回原始参数空间
    ci_laplace_original = zeros(3,2);
    ci_laplace_original(1,:) = ci_laplace(1,:).^2;
    ci_laplace_original(2,:) = ci_laplace(2,:).^2;
    ci_laplace_original(3,:) = 1 ./ (1 + exp(-ci_laplace(3,:)));
    
    total_time = toc;
    
    % 返回置信区间
    ci_laplace = ci_laplace_original;
end