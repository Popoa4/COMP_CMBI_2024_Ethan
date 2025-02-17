function sumRes = BallStickSSD_transformed(x, Avox, bvals, qhat)
    % 变换参数以符合物理约束
    S0 = x(1)^2;
    d = x(2)^2;
    f = 1 / (1 + exp(-x(3)));
    theta = x(4);
    phi = x(5);
    
    % 计算纤维方向（与之前相同）
    fibdir = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
    fibdotgrad = sum(qhat .* repmat(fibdir, [size(qhat,1), 1]), 2);
    
    % 计算模型信号
    S = S0 * ( f * exp(-bvals*d .* (fibdotgrad.^2)) + (1-f) * exp(-bvals*d) );
    
    % 计算残差平方和
    sumRes = sum((Avox - S).^2);
end
