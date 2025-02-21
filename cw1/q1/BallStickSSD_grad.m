function [sumRes, grad] = BallStickSSD_grad(x, Avox, bvals, qhat)
    S0 = x(1);
    d = x(2);
    f = x(3);
    theta = x(4);
    phi = x(5);
    
    fibdir = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
    fibdotgrad = sum(qhat .* fibdir, 2); 
    
    E_stick = exp(-bvals*d.*fibdotgrad.^2);
    E_ball = exp(-bvals*d);
    S = S0*(f*E_stick + (1-f)*E_ball);
    
    residuals = Avox - S;
    sumRes = sum(residuals.^2);
    
    % ===== 梯度计算修正 =====
    % 公共项
    common_term = -2 * residuals; % 注意负号
    
    % 1. S0的梯度
    dS_dS0 = (f*E_stick + (1-f)*E_ball);
    grad_S0 = sum(common_term .* dS_dS0);
    
    % 2. d的梯度
    dE_dd_stick = -bvals .* fibdotgrad.^2 .* E_stick;
    dE_dd_ball = -bvals .* E_ball;
    dS_dd_part1 = S0 * f * dE_dd_stick;
    dS_dd_part2 = S0 * (1-f) * dE_dd_ball;
    dS_dd = dS_dd_part1 + dS_dd_part2; 

    grad_d = sum(common_term .* dS_dd);

    
    % 3. f的梯度
    dS_df = S0 * (E_stick - E_ball);
    grad_f = sum(common_term .* dS_df);
    
    % 4. theta的梯度
    dfibdir_dtheta = [cos(phi)*cos(theta), sin(phi)*cos(theta), -sin(theta)];
    dfibdotgrad_dtheta = sum(qhat .* dfibdir_dtheta, 2);
    dE_dtheta = -bvals*d .* 2.*fibdotgrad .* dfibdotgrad_dtheta .* E_stick;
    dS_dtheta = S0*f * dE_dtheta;
    grad_theta = sum(common_term .* dS_dtheta);
    
    % 5. phi的梯度
    dfibdir_dphi = [-sin(phi)*sin(theta), cos(phi)*sin(theta), 0];
    dfibdotgrad_dphi = sum(qhat .* dfibdir_dphi, 2);
    dE_dphi = -bvals*d .* 2.*fibdotgrad .* dfibdotgrad_dphi .* E_stick;
    dS_dphi = S0*f * dE_dphi;
    grad_phi = sum(common_term .* dS_dphi);
    
    % 组合梯度向量
    grad = [grad_S0, grad_d, grad_f, grad_theta, grad_phi];
end
