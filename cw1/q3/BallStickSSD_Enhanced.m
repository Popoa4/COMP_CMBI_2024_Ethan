function sumRes = BallStickSSD_Enhanced(x, meas, bvals, grad_dirs)
    % 直接使用原始参数
    S0 = x(1);   % 需约束 S0 > 0
    d = x(2);    % 需约束 d > 0
    f = x(3);    % 需约束 0 ≤ f ≤ 1
    theta = x(4);% 需约束 0 ≤ theta ≤ π
    phi = x(5);  % 需约束 0 ≤ phi ≤ 2π
    
    % 计算纤维方向
    fib_dir = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
    
    % 计算信号
    fib_dot_grad = sum(grad_dirs .* fib_dir, 2);
    S_pred = S0 * (f * exp(-bvals.*d.*fib_dot_grad.^2) + (1-f)*exp(-bvals.*d));
    
    % 残差平方和
    sumRes = sum((meas - S_pred).^2);
end