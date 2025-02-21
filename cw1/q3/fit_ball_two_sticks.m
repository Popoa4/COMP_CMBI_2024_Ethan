function [params_opt, resnorm] = fit_ball_two_sticks(meas, bvals, grad_dirs)
    % Parameter order: [S0, d, f1, theta1, phi1, f2, theta2, phi2]
    % Constraints: S0>0, d>0, 0≤f1,f2≤1, f1+f2≤1, angular periodicity
    [S0_init, d_init, theta_init, phi_init] = dti_initialization(meas, bvals, grad_dirs);
    x0 = [S0_init, trace(d_init)/3, 0.6, theta_init, phi_init, 0.4, theta_init+ pi/4, phi_init+ pi/4];

    lb = [0,   0,   0,   0,    0,   0,    0,     0];
    ub = [Inf, Inf, 1,   pi, 2*pi, 1,   pi,  2*pi];

    % Nonlinear constraints (f1+f2 ≤ 1)
    function [c, ceq] = nonlcon(x)
        c = x(3) + x(6) - 1; 
        ceq = [];
    end

    options = optimoptions('fmincon', 'Algorithm','sqp', 'Display','off');
    num_trials = 50;
    min_resnorm = Inf;
    for i = 1:num_trials
        pert = [0.5*x0(1), 0.5*x0(2), 0.5*x0(3), 0.5*x0(4), 0.5*x0(5), 0.5*x0(6), 0.5*x0(7), 0.5*x0(8)] .* randn(1,8);
        x_init = max(lb, min(ub, x0 + pert));
        [params, resnorm] = fmincon(@(x)ball_two_sticks_obj(x,meas,bvals,grad_dirs),...
                               x_init, [],[],[],[],lb,ub,@nonlcon,options);
        if resnorm < min_resnorm
            min_resnorm = resnorm;
            params_opt = params;
        end
    end
    resnorm = min_resnorm;
    
    function sumRes = ball_two_sticks_obj(x, meas, bvals, grad_dirs)
        S0 = x(1);
        d = x(2);
        f1 = x(3);
        theta1 = x(4);
        phi1 = x(5);
        f2 = x(6);
        theta2 = x(7);
        phi2 = x(8);

        n1 = [sin(theta1)*cos(phi1), sin(theta1)*sin(phi1), cos(theta1)];
        n2 = [sin(theta2)*cos(phi2), sin(theta2)*sin(phi2), cos(theta2)];

        dot_prod1 = grad_dirs * n1';
        dot_prod2 = grad_dirs * n2';
        S_stick1 = exp(-bvals * d .* dot_prod1.^2);
        S_stick2 = exp(-bvals * d .* dot_prod2.^2);
        S_ball = exp(-bvals * d);
        S_pred = S0 * (f1*S_stick1 + f2*S_stick2 + (1-f1-f2)*S_ball);

        sumRes = sum((meas - S_pred).^2);
    end
end

