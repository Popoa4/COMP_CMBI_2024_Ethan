function [params, resnorm] = fit_ball_multi_sticks(meas, bvals, grad_dirs, num_sticks)
    % meas: 测量信号
    % bvals: b值
    % grad_dirs: 梯度方向
    % num_sticks: Stick分量数量

    % 参数数量
    % num_sticks = 5;
    num_params = 3*num_sticks + 2;

    % 参数顺序: [S0, d, f1, theta1, phi1, f2, theta2, phi2, ..., fN, thetaN, phiN]
    
    % 初始化策略 (基于Zeppelin and Stick, 结合随机扰动)
     % [~, zs_params] = fit_zeppelin_stick(meas, bvals, grad_dirs);
    [S0_init, d_init, theta_init, phi_init] = dti_initialization(meas, bvals, grad_dirs);
    
    S0_init = S0_init;
    d_init = trace(d_init)/3;
    
    % 初始分配各stick的体积分数
    f_init = zeros(1,num_sticks);
    for i= 1:num_sticks
        f_init(i) = 0.5 + randn()*0.05; 
    end

    % 初始角度 (在Zeppelin主方向附近随机扰动)
    angles_init = zeros(2, num_sticks);
    for i = 1:num_sticks
        angles_init(1, i) = theta_init + randn() * 0.2; % theta
        angles_init(2, i) = phi_init + randn() * 0.2; % phi
    end
    
    % 合并初始参数
     x0 = [S0_init, d_init, f_init, angles_init(:)'];

    % 边界约束
    lb = [0, 0, zeros(1, num_sticks), zeros(1, 2*num_sticks)];          % S0, d, f_i > 0
    ub = [Inf, Inf, ones(1, num_sticks), pi*ones(1,num_sticks), 2*pi*ones(1,num_sticks)]; % f_i < 1, 角度范围

    % 非线性约束 (所有体积分数之和小于1)
    function [c, ceq] = nonlcon(x)
        c = sum(x(3:3+num_sticks-1)) - 1; % f1 + f2 + ... + fN ≤ 1
        ceq = [];
    end

    % 优化选项
    options = optimoptions('fmincon', 'Algorithm','sqp', 'Display','off',...
                           'MaxFunctionEvaluations', 10000);

    % 执行优化
    [params, resnorm] = fmincon(@(x)ball_multi_sticks_obj(x,meas,bvals,grad_dirs, num_sticks),...
                               x0, [],[],[],[],lb,ub,@nonlcon,options);

    function sumRes = ball_multi_sticks_obj(x, meas, bvals, grad_dirs, num_sticks)
        S0 = x(1);
        d = x(2);
        f = x(3:2+num_sticks); % 各stick体积分数
        angles = reshape(x(3+num_sticks:end), 2, num_sticks); % 角度参数

        % 计算信号
        S_ball = S0 * (1 - sum(f)) * exp(-bvals * d); % Ball分量
        S_sticks = zeros(size(meas));
        
        for i = 1:num_sticks
            theta = angles(1, i);
            phi = angles(2, i);
            n = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]; % 方向向量
            dot_prod = grad_dirs * n';
            S_sticks = S_sticks + S0 * f(i) * exp(-bvals * d .* dot_prod.^2); % 各Stick分量
        end

        S_pred = S_ball + S_sticks;
        sumRes = sum((meas - S_pred).^2);
    end
end
