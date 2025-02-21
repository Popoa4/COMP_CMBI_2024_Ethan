function [ci_bootstrap, total_time] = bootstrap_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y, num_bootstrap)
    tic;
    
    Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));
    num_measurements = length(Avox_original); % 108
    
    [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    
    % Preallocate
    params_bootstrap = zeros(num_bootstrap, 3); % [S0, d, f]
    
    fprintf('Starting classic bootstrap estimation, %d times in total...\n', num_bootstrap);
    
    parfor i = 1:num_bootstrap
        indices = randi(num_measurements, num_measurements, 1);
        Avox_boot = Avox_original(indices);
        qhat_boot = qhat(indices, :);
        bvals_boot = bvals(indices);
        Y_boot = build_design_matrix(bvals_boot, qhat_boot);
        
        [S0_bs, d_bs, f_bs, ~] = robust_ball_stick_fit(Avox_boot, bvals_boot, qhat_boot, Y_boot, S0_init, d_init, f_init, theta_init, phi_init);
        
        params_bootstrap(i, :) = [S0_bs, d_bs, f_bs];
    end
    
    % Computing confidence intervals (2-sigma ~ 95%)
    ci_bootstrap = prctile(params_bootstrap, [2.5 97.5]);
    total_time = toc;
end
