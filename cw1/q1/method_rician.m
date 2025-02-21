function [results, total_time] = method_rician(dwis, bvals, qhat, slice_num)
    tic;
    [x_dim, y_dim] = size(dwis, 2:3);
    
    % 初始参数设置（与之前相同）
    startx_original = [3.5e3, 3e-3, 0.25, 0, 0, 0.5]; % 增加sigma初始值
    S0_init = startx_original(1)* ones(x_dim, y_dim);
    d_init = startx_original(2)* ones(x_dim, y_dim);
    f_init = startx_original(3)* ones(x_dim, y_dim);
    theta_init = startx_original(4)* ones(x_dim, y_dim);
    phi_init = startx_original(5)* ones(x_dim, y_dim);
    sigma_init = startx_original(6) * ones(x_dim, y_dim); % 初始化sigma
    
    % 设置信噪比阈值
    snr_threshold = 3;
    h = optimset('Algorithm','quasi-newton', 'Display','off', 'MaxFunEvals',1e4);
    S0 = zeros(x_dim, y_dim);
    d = zeros(x_dim, y_dim);
    f = zeros(x_dim, y_dim);
    RESNORM = zeros(x_dim, y_dim);
    theta = zeros(x_dim, y_dim);
    phi = zeros(x_dim, y_dim);
    sigma = zeros(x_dim,y_dim); % 存储sigma

    for x = 1:x_dim
        for y = 1:y_dim
            Avox = squeeze(dwis(:,x,y,slice_num));
            if all(Avox>0)
                 startx = [sqrt(S0_init(x,y)), sqrt(d_init(x,y)),...
                     -log(1/f_init(x,y)-1), theta_init(x,y), phi_init(x,y), sigma_init(x,y)]; % 初始值包含sigma
                try
                    % 使用fminunc和新的目标函数
                    [params, resnorm] = fminunc(@(x)BallStickRicianNLL(x,Avox,bvals,qhat, snr_threshold), startx, h);
                    S0(x,y) = params(1)^2;
                    d(x,y) = params(2)^2;
                    f(x,y) = 1/(1+exp(-params(3)));
                    theta(x,y) = params(4);
                    phi(x,y) = params(5);
                    sigma(x,y) = params(6); % 优化sigma
                    RESNORM(x,y) = resnorm;
                catch
                    RESNORM(x,y) = NaN;
                end
            end
        end
    end
    
    % 存储结果
        results = struct('S0',S0, 'd',d, 'f',f,...
                    'RESNORM',RESNORM, 'theta',theta, 'phi',phi, 'sigma', sigma);
    total_time = toc;
end