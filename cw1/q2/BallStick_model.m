function S = BallStick_model(x, bvals, qhat)
    S0 = x(1);
    d = x(2);
    f = x(3);
    theta = x(4);
    phi = x(5);
    
    % 计算纤维方向
    fibdir = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
    
    % 计算纤维方向与梯度方向的点积
    fibdotgrad = sum(qhat .* repmat(fibdir, [size(qhat,1),1]), 2);
    
    % 计算模型信号
    S = S0 * (f * exp(-bvals*d .* (fibdotgrad.^2)) + (1-f) * exp(-bvals*d));
end