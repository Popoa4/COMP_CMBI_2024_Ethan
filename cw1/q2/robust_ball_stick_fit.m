function [S0_bs, d_bs, f_bs, resnorm_best] = robust_ball_stick_fit(Avox, bvals, qhat, Y, S0_init, d_init, f_init, theta_init, phi_init)
    num_trials = 5; % 拟合次数
    tolerance = 1e-6;
    perturb_scales = [0.05*sqrt(S0_init), 0.05*sqrt(d_init), 0.1, 0.05*pi, 0.05*2*pi];
    
    params_all = zeros(num_trials, 5);
    resnorm_all = Inf(num_trials, 1);
    
    for i = 1:num_trials
        % 生成扰动后的起始点
        perturbation = randn(1,5) .* perturb_scales;
        startx_transformed = [sqrt(S0_init), sqrt(d_init), -log((1/f_init)-1), theta_init, phi_init] + perturbation;
        
        % 执行拟合
        try
            [params_hat_trans, resnorm] = fminunc(@(x) BallStickSSD_transformed(x, Avox, bvals, qhat), ...
                                                 startx_transformed, ...
                                                 optimset('MaxFunEvals',20000, 'Algorithm','quasi-newton', ...
                                                         'TolX',1e-10, 'TolFun',1e-10, 'Display','off'));
            resnorm_all(i) = resnorm;
            params_all(i, :) = params_hat_trans;
        catch
            resnorm_all(i) = Inf;
            params_all(i, :) = startx_transformed;
        end
    end
    
    % 选择最佳拟合结果
    [resnorm_best, best_idx] = min(resnorm_all);
    best_params_trans = params_all(best_idx, :);
    
    % 反变换回原始参数空间
    S0_bs = best_params_trans(1)^2;
    d_bs = best_params_trans(2)^2;
    f_bs = 1 / (1 + exp(-best_params_trans(3)));
end