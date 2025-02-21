function [ci_laplace, total_time] = laplace_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y)
    tic;
    Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    
    % Fitting raw data, obtaining parameter estimates
    [S0_est, d_est, f_est, ~] = robust_ball_stick_fit(Avox_original, bvals, qhat, Y, S0_init, d_init, f_init, theta_init, phi_init);
    
    S0_trans = sqrt(S0_est);
    d_trans = sqrt(d_est);
    f_trans = -log((1/f_est) - 1);
    theta_trans = theta_init;
    phi_trans = phi_init;
    params_trans = [S0_trans, d_trans, f_trans, theta_trans, phi_trans];

    options = optimoptions('fminunc', ...
        'Display', 'off', ...
        'Algorithm', 'quasi-newton', ...
        'SpecifyObjectiveGradient', true);
    
    % Perform fit and get Hessian matrix
    [params_hat_trans, ~, ~, ~, ~, Hessian] = fminunc(@(x) BallStickSSD_grad(x, Avox_original, bvals, qhat), params_trans, options);

    % Compute covariance matrix (inverse of Hessian)
    cov_matrix = inv(Hessian);
    
    % Confidence intervals (95%)
    std_devs = sqrt(diag(cov_matrix(1:3, 1:3))); % Extract variances only for S0, d, f
    
    ci_laplace = zeros(3,2);
    ci_laplace(:, 1) = params_hat_trans(1:3)' - 1.96 * std_devs; 
    ci_laplace(:, 2) = params_hat_trans(1:3)' + 1.96 * std_devs; 
    
    % Back transform back to original parameter space
    ci_laplace_original = zeros(3,2);
    ci_laplace_original(1,:) = ci_laplace(1,:).^2;
    ci_laplace_original(2,:) = ci_laplace(2,:).^2;
    ci_laplace_original(3,:) = 1 ./ (1 + exp(-ci_laplace(3,:)));
    
    total_time = toc;

    ci_laplace = ci_laplace_original;
end