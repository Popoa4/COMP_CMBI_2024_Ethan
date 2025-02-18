% 使用DTI初始点 + fmincon (约束优化)
function [results, total_time] = method_fmincon(dwis, bvals, qhat, slice_num)
    tic;
    [x_dim, y_dim] = size(dwis, 2:3);
    % 步骤1: 计算DTI参数
    % [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization(dwis, bvals, qhat, slice_num);
    
     h = optimoptions('fmincon', ...
                     'Algorithm', 'sqp', ...  % 更稳健的算法
                     'Display', 'off', ...
                     'MaxFunctionEvaluations', 20000, ...
                     'MaxIterations', 1000, ...
                     'OptimalityTolerance', 1e-6, ...
                     'StepTolerance', 1e-6);
    % h = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                     % 'SpecifyObjectiveGradient', true, 'Display', 'off', ...
                     % 'MaxFunctionEvaluations', 1e4, 'Tolfun', 1e-10, 'TolX', 1e-10);

    lb = [0 0 0 0 0];       % S0, d, f, theta, phi下限
    % ub = [1e5 3e-3 1 pi 2*pi]; % 上限
    ub = [Inf, Inf, 1, pi, 2*pi];   
    
    % 预分配结果矩阵
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
                % startx = [mean(Avox(bvals==0)), 2e-3, 0.3, pi/4, pi/4]; % 通用初始值
                startx = [3.5e3, 3e-3, 0.25, 0, 0];
                % startx = [S0_init(x,y), d_init(x,y),...
                         % f_init(x,y), theta_init(x,y), phi_init(x,y)];
                try
                    % 设置超时限制（100秒/体素）
                    % opt = h; 
                    % opt.TimeLimit = 100;
                    [params, resnorm] = fmincon(@(x)BallStickSSD(x,Avox,bvals,qhat),...
                                              startx, [],[],[],[],lb,ub,[],h);
                    % [S0(x,y), d(x,y), f(x,y), theta(x,y), phi(x,y)] = parse_params(params);
                    S0(x,y) = params(1);
                    d(x,y) = params(2);
                    f(x,y) = params(3);
                    theta(x,y) = params(4);
                    phi(x,y) = params(5);
                    RESNORM(x,y) = resnorm;
                catch ME
                    fprintf('体素(%d,%d) 优化失败: %s\n',x,y,ME.message);
                    RESNORM(x,y) = NaN;
                end
            else
                % 无效信号体素
                S0(x,y) = NaN;
                d(x,y) = NaN;
                f(x,y) = NaN;
                RESNORM(x,y) = NaN;
                theta(x,y) = NaN;
                phi(x,y) = NaN;
            end
        end
    end
    
    results = struct('S0',S0, 'd',d, 'f',f,...
                    'RESNORM',RESNORM, 'theta',theta, 'phi',phi);
    total_time = toc;
end