function sumRes = BallStickSSD_Enhanced(x, meas, bvals, grad_dirs)
    S0 = x(1);   
    d = x(2);    
    f = x(3);    
    theta = x(4);
    phi = x(5);  
    
    % Calculate fiber orientation
    fib_dir = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
    
    fib_dot_grad = sum(grad_dirs .* fib_dir, 2);
    S_pred = S0 * (f * exp(-bvals.*d.*fib_dot_grad.^2) + (1-f)*exp(-bvals.*d));
    sumRes = sum((meas - S_pred).^2);
end