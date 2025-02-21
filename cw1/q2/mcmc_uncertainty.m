function [ci_95_MCMC, total_time] = mcmc_uncertainty(dwis, bvals, qhat, slice_num, voxel_coords, Y)
    tic;
    Avox_original = squeeze(dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3)));

    % DTI初始化获取起始点
    [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization_single_voxel(Y, Avox_original);
    theta_init = mod(theta_init, pi);
    phi_init = mod(phi_init, 2*pi);
    %% Defining MCMC parameters
    num_iterations = 10000;    
    burn_in = 2000;            
    thinning = 10;             
    lb = [0, 0, 0, 0, 0];      
    ub = [Inf, Inf, 1, pi, 2*pi]; 
    
    current_params = [S0_init, d_init, f_init, theta_init, phi_init];
    
    sigma_noise = 200;
    log_likelihood = @(x) -0.5 * sum((Avox_original - BallStick_model(x, bvals, qhat)).^2) / sigma_noise^2;
    
    % Posterior distribution (logarithmic)
    log_posterior = @(x) log_prior(x) + log_likelihood(x);
    
    %% Define proposal distribution (Gaussian perturbation）
    proposal_std = [100, 1e-4, 0.05, 0.1, 0.1];
    proposal = @(x) x + proposal_std .* randn(1,5);
    
    samples = zeros(num_iterations, 5);
    acceptance = 0;
    
    %% MCMCSampling loop
    fprintf('Start MCMC sampling...\n');
    for i = 1:num_iterations
        lp_current = log_posterior(current_params);
        if isnan(lp_current)
            fprintf('Iteration %d: current_params prior=-Inf!\n', i);
            return;
        end
        proposed_params = proposal(current_params);
        
        % Calculate posterior probability
        log_posterior_current = log_posterior(current_params);
        log_posterior_proposed = log_posterior(proposed_params);
        if isnan(log_posterior_current)
            disp('Iteration %d:log_posterior_current is NaN!', i);
            return;
        elseif isinf(log_posterior_current)
            disp('log_posterior_current is Inf!');
        end
        if isnan(log_posterior_proposed)
            disp('Iteration %d:log_posterior_proposed is NaN!', i);
            return;
        elseif isinf(log_posterior_proposed)
            disp('log_posterior_proposed is Inf!');
        end
        
        % Calculate acceptance rate
        log_accept_ratio = log_posterior_proposed - log_posterior_current;
        if log(rand) < log_accept_ratio
            current_params = proposed_params;
            acceptance = acceptance + 1;
        end
        samples(i, :) = current_params;
    end
    
    %% Post-processing
    % Calculate acceptance rate
    acceptance_rate = acceptance / num_iterations;
    fprintf('MCMC acceptance rate: %.4f%%\n', acceptance_rate * 100);
    
    % Discard burn-in period and perform thinningDiscard burn-in period and perform thinning
    samples_post = samples(burn_in:thinning:end, :);
    
    %% Calculate confidence interval
    ci_95_MCMC = prctile(samples_post, [2.5 97.5]);

    total_time = toc;
end

function lp = log_prior(x)
    if x(1)>0 && x(2)>0 && x(3)>=0 && x(3)<=1 && ...
       x(4)>=0 && x(4)<=pi && x(5)>=0 && x(5)<=2*pi
        lp = 0;     
    else
        lp = -Inf;    
    end
end


