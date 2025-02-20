% 0. 使用设定初始点 + 参数变换 + fminunc
function [results, total_time] = method_origin(dwis, bvals, qhat, slice_num)
    tic;
    [x_dim, y_dim] = size(dwis, 2:3);
    
    % 步骤1: 初始参数
    % [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization(dwis, bvals, qhat, slice_num);

    startx_original = [3.5e3, 3e-3, 0.25, 0, 0];
    S0_init = startx_original(1)* ones(x_dim, y_dim);
    d_init = startx_original(2)* ones(x_dim, y_dim);
    f_init = startx_original(3)* ones(x_dim, y_dim);
    theta_init = startx_original(4)* ones(x_dim, y_dim);
    phi_init = startx_original(5)* ones(x_dim, y_dim);
    
    % 步骤2: 球棍模型拟合
    % [S0_bs, d_bs, f_bs, RESNORM, theta_bs, phi_bs] = ball_stick_fitting(...
    %     dwis, bvals, qhat, slice_num, S0_map, d_map, f_map, theta_map, phi_map);

    h = optimset('Algorithm','quasi-newton', 'Display','off', 'MaxFunEvals',1e4);
    S0 = zeros(x_dim, y_dim);
    d = zeros(x_dim, y_dim);
    f = zeros(x_dim, y_dim);
    RESNORM = zeros(x_dim, y_dim);
    theta = zeros(x_dim, y_dim);
    phi = zeros(x_dim, y_dim);
    
    for x = 1:x_dim
        for y = 1:y_dim
            Avox = squeeze(dwis(:,x,y,slice_num));
            if all(Avox>0)
                startx = [sqrt(S0_init(x,y)), sqrt(d_init(x,y)),...
                         -log(1/f_init(x,y)-1), theta_init(x,y), phi_init(x,y)];
                try
                    [params, resnorm] = fminunc(@(x)BallStickSSD_transformed(x,Avox,bvals,qhat), startx, h);
                    S0(x,y) = params(1)^2;
                    d(x,y) = params(2)^2;
                    f(x,y) = 1/(1+exp(-params(3)));
                    theta(x,y) = params(4);
                    phi(x,y) = params(5);
                    RESNORM(x,y) = resnorm;
                catch
                    RESNORM(x,y) = NaN;
                end
            end
        end
    end
    % 存储结果
    results = struct('S0',S0, 'd',d, 'f',f,...
                    'RESNORM',RESNORM, 'theta',theta, 'phi',phi);
    total_time = toc;
end