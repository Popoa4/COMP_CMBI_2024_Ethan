function [ci_param_bootstrap, total_time] = parametric_bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, num_bootstrap)
    tic;
    Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    
    [S0_est, d_est, f_est, ~] = robust_ball_stick_fit(Avox_original, bvals, qhat, Y, S0_init, d_init, f_init, theta_init, phi_init);

    % Generate samples
    params_bootstrap = zeros(num_bootstrap, 3);
    
    % Precompute fiber directions
    fibdir = [cos(phi_init)*sin(theta_init), sin(phi_init)*sin(theta_init), cos(theta_init)];
    fibdotgrad = sum(qhat .* fibdir, 2);
    fprintf('Start parametric bootstrap estimation, %d times...\n', num_bootstrap);
    
    parfor i = 1:num_bootstrap
        % a Generate synthetic data
        S_model = S0_est * ( f_est * exp(-bvals .* d_est .* (fibdotgrad.^2)) ...
                           + (1 - f_est) * exp(-bvals .* d_est) );
        noise    = randn(size(S_model)) * 200;
        Avox_boot= S_model + noise;
        Avox_boot(Avox_boot < 1e-10) = 1e-10;
        [S0_bs, d_bs, f_bs, ~] = robust_ball_stick_fit(Avox_boot, bvals, qhat, Y, S0_init, d_init, f_init, theta_init, phi_init);

        params_bootstrap(i, :) = [S0_bs, d_bs, f_bs];
    end
    
    % Calculate confidence interval (2-sigma ~ 95%)
    ci_param_bootstrap = prctile(params_bootstrap, [2.5 97.5]);
    total_time = toc;
end
