function [params, resnorm] = fit_ball_multi_sticks(meas, bvals, grad_dirs, num_sticks)
    num_params = 3*num_sticks + 2;
    % Parameter order: [S0, d, f1, theta1, phi1, f2, theta2, phi2, ..., fN, thetaN, phiN]
    [S0_init, d_init, theta_init, phi_init] = dti_initialization(meas, bvals, grad_dirs);
    
    S0_init = S0_init;
    d_init = trace(d_init)/3;
    
    % Initially assign volume fractions to each stick
    f_init = zeros(1,num_sticks);
    for i= 1:num_sticks
        f_init(i) = 0.5 + randn()*0.05; 
    end

    % Initial angle (random perturbation around the main direction of Zeppelin)
    angles_init = zeros(2, num_sticks);
    for i = 1:num_sticks
        angles_init(1, i) = theta_init + randn() * 0.2; % theta
        angles_init(2, i) = phi_init + randn() * 0.2; % phi
    end
    
    % Merge initial parameters
     x0 = [S0_init, d_init, f_init, angles_init(:)'];
    lb = [0, 0, zeros(1, num_sticks), zeros(1, 2*num_sticks)];       
    ub = [Inf, Inf, ones(1, num_sticks), pi*ones(1,num_sticks), 2*pi*ones(1,num_sticks)]; 

    % Nonlinear constraints (the sum of all volume fractions is less than 1)
    function [c, ceq] = nonlcon(x)
        c = sum(x(3:3+num_sticks-1)) - 1; 
        ceq = [];
    end

    options = optimoptions('fmincon', 'Algorithm','sqp', 'Display','off',...
                           'MaxFunctionEvaluations', 10000);
    [params, resnorm] = fmincon(@(x)ball_multi_sticks_obj(x,meas,bvals,grad_dirs, num_sticks),...
                               x0, [],[],[],[],lb,ub,@nonlcon,options);

    function sumRes = ball_multi_sticks_obj(x, meas, bvals, grad_dirs, num_sticks)
        S0 = x(1);
        d = x(2);
        f = x(3:2+num_sticks); 
        angles = reshape(x(3+num_sticks:end), 2, num_sticks); 

        % Calculating Signals
        S_ball = S0 * (1 - sum(f)) * exp(-bvals * d); 
        S_sticks = zeros(size(meas));
        
        for i = 1:num_sticks
            theta = angles(1, i);
            phi = angles(2, i);
            n = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)];
            dot_prod = grad_dirs * n';
            S_sticks = S_sticks + S0 * f(i) * exp(-bvals * d .* dot_prod.^2);
        end

        S_pred = S_ball + S_sticks;
        sumRes = sum((meas - S_pred).^2);
    end
end
