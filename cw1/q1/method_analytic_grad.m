% 使用DTI初始点 + fmincon + 解析梯度
function [results, total_time] = method_analytic_grad(dwis, bvals, qhat, slice_num)
    tic;
    [x_dim, y_dim] = size(dwis, 2:3);
    % 步骤1: 计算DTI参数
    [S0_init, d_init, f_init, theta_init, phi_init] = dti_initialization(dwis, bvals, qhat, slice_num);
    
    % h = optimset('Algorithm','sqp', 'Display','off',...
    %             'MaxFunEvals',1e4,...
    %             'TolFun',1e-8, 'TolX',1e-8,...
    %             'Display','off');
    
    h = optimoptions('fminunc', 'Algorithm', 'quasi-newton', ...
                     'SpecifyObjectiveGradient', true, 'Display', 'off', ...
                     'MaxFunctionEvaluations', 1e4, 'Tolfun', 1e-10, 'TolX', 1e-10);

    lb = [0 0 0 0 0];       % S0, d, f, theta, phi下限
    ub = [1e5 3e-3 1 pi 2*pi]; % 上限
    
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
                startx = [sqrt(S0_init(x,y)), sqrt(d_init(x,y)),...
                         -log(1/f_init(x,y)-1), theta_init(x,y), phi_init(x,y)];
                try
                    % 设置超时限制（100秒/体素）
                    opt = h; 
                    opt.TimeLimit = 100;
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
                    continue;
                end
            end
        end
    end
    
    results = struct('S0',S0, 'd',d, 'f',f,...
                    'RESNORM',RESNORM, 'theta',theta, 'phi',phi);
    total_time = toc;
end