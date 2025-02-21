function [params_opt, resnorm] = fit_tortuosity(meas, bvals, grad_dirs)
    % Parameter order: [S0, d, f, theta, phi]
    lb = [0,   0,   0,   0,    0];
    ub = [Inf, Inf, 1,   pi, 2*pi];
    
    % DTI-based initialization
    [S0_init, d_init, theta_init, phi_init] = dti_initialization(meas, bvals, grad_dirs);
    x0 = [S0_init, trace(d_init)/3, 0.5, theta_init, phi_init];  
    options = optimoptions('fmincon','Algorithm','sqp','Display','off');

    num_trials = 100;
    min_resnorm = Inf;
    for i = 1:num_trials
        % pert = [0.2*S0_init, 0.2*max(lambda), 0.3, 0.5*pi, pi].*randn(1,5);
        pert = [0.5*x0(1), 0.5*x0(2), 0.5*x0(3), 0.5*pi, pi].*randn(1,5);
        x_init = max(lb, min(ub, x0 + pert));
        [x, res] = fmincon(@(x)tortuosity_obj(x,meas,bvals,grad_dirs),...
                          x_init, [],[],[],[],lb,ub,[],options);
        if res < min_resnorm
            min_resnorm = res;
            params_opt = x;
        end
    end
    resnorm = min_resnorm;
    
    function sumRes = tortuosity_obj(x, meas, bvals, grad_dirs)
        S0 = x(1);
        lambda1 = x(2);
        f = x(3);
        theta = x(4);
        phi = x(5);
        lambda2 = (1-f)*lambda1;
        
        n = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)];

        dot_prod = grad_dirs * n';
        S_stick = exp(-bvals*lambda1.*dot_prod.^2);
        S_zeppelin = exp(-bvals.*(lambda2 + (lambda1-lambda2).*dot_prod.^2));
        S_pred = S0*(f*S_stick + (1-f)*S_zeppelin);
        
        sumRes = sum((meas - S_pred).^2);
    end
end
