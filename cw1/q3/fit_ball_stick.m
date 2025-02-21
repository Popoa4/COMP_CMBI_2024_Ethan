function [params_opt, resnorm] = fit_ball_stick(meas, bvals, grad_dirs)
    % Parameters: [S0, d, f, theta, phi]
    lb = [0,   0,   0,   0,    0];
    ub = [Inf, Inf, 1,   pi, 2*pi];
    
    % DTI-based initialization
    [S0_init, d_init, theta_init, phi_init] = dti_initialization(meas, bvals, grad_dirs);
    x0 = [S0_init, trace(d_init)/3, 0.5, theta_init, phi_init];
    options = optimoptions('fmincon','Algorithm','sqp','Display','off');
    num_trials = 50;
    min_resnorm = Inf;
    for i = 1:num_trials
        pert = [0.2*x0(1), 0.2*x0(2), 0.3, 0.5*pi, pi] .* randn(1,5);
        x_init = max(lb, min(ub, x0 + pert));
        [x, res] = fmincon(@(x)ball_stick_obj(x,meas,bvals,grad_dirs),...
                          x_init, [],[],[],[],lb,ub,[],options);
        if res < min_resnorm
            min_resnorm = res;
            params_opt = x;
        end
    end
    resnorm = min_resnorm;
    
    % Objective function
    function sumRes = ball_stick_obj(x, meas, bvals, grad_dirs)
        S0 = x(1); d = x(2); f = x(3);
        theta = x(4); phi = x(5);
        n = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)];
        S_pred = S0*(f*exp(-bvals*d.*(grad_dirs*n').^2) + (1-f)*exp(-bvals*d));
        sumRes = sum((meas - S_pred).^2);
    end
end
