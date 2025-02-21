function [params_opt, resnorm] = fit_zeppelin_stick(meas, bvals, grad_dirs)
    % Parameter order: [S0, d, f, theta, phi, lambda2]
    % Constraints: S0>0, d≥lambda2≥0, 0≤f≤1
    [S0_init, d_init, theta_init, phi_init] = dti_initialization(meas, bvals, grad_dirs);
    lambda2_init = trace(d_init)/3 * 0.5;

    x0 = [S0_init, trace(d_init)/3, 0.5, theta_init, phi_init, lambda2_init];
    lb = [0,   0,   0,   0,    0,    0];
    ub = [Inf, Inf, 1,   pi, 2*pi, Inf];

    % Nonlinear constraints (d ≥ lambda2)
    function [c, ceq] = nonlcon(x)
        c = x(6) - x(2);
        ceq = [];
    end
    options = optimoptions('fmincon', 'Algorithm','sqp', 'Display','off');

    num_trials = 100;
    min_resnorm = Inf;
    for i = 1:num_trials
        pert = [0.5*x0(1), 0.5*x0(2), 0.5*x0(3), 0.5*x0(4), 0.5*x0(5), 0.5*x0(6)].*randn(1,6);
        x_init = max(lb, min(ub, x0 + pert));
        [x, res] = fmincon(@(x)zeppelin_stick_obj(x,meas,bvals,grad_dirs),...
                               x_init, [],[],[],[],lb,ub,@nonlcon,options);
    
        if res < min_resnorm
            min_resnorm = res;
            params_opt = x;
        end
    end
    resnorm = min_resnorm;
    
    % Objective Function
    function sumRes = zeppelin_stick_obj(x, meas, bvals, grad_dirs)
        S0 = x(1);
        lambda1 = x(2);
        f = x(3);
        theta = x(4);
        phi = x(5);
        lambda2 = x(6);
        
        n = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)];
        dot_prod = grad_dirs * n';
        S_stick = exp(-bvals*lambda1.*dot_prod.^2);
        S_zeppelin = exp(-bvals.*(lambda2 + (lambda1-lambda2).*dot_prod.^2));
        S_pred = S0*(f*S_stick + (1-f)*S_zeppelin);
        
        sumRes = sum((meas - S_pred).^2);
    end
end
