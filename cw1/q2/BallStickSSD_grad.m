function [sumRes, grad] = BallStickSSD_grad(x, Avox, bvals, qhat)
    % 参数变换
    S0 = x(1)^2;
    d = x(2)^2;
    f = 1 / (1 + exp(-x(3)));
    theta = x(4);
    phi = x(5);

    % 计算纤维方向
    fibdir = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
    fibdotgrad = sum(qhat .* repmat(fibdir, [size(qhat,1), 1]), 2); % 正确使用按元素乘法

    % 计算模型信号
    E_stick = exp(-bvals * d .* (fibdotgrad.^2)); % 注意: bvals 是列向量，fibdotgrad.^2 是列向量
    E_ball = exp(-bvals * d);
    S = S0 * (f * E_stick + (1-f) * E_ball);
    residuals = Avox - S;
    sumRes = sum(residuals.^2);

    % 计算梯度 (按元素操作使用 .*)
    % --- dS0_trans ---
    dS0_trans = -4 * x(1) * sum(residuals .* (f * E_stick + (1-f) * E_ball));

    % --- dd_trans ---
    term1 = f * (-bvals .* fibdotgrad.^2) .* E_stick; % 修正为按元素乘法 .*
    term2 = (1-f) * (-bvals) .* E_ball;
    dd_trans = -4 * x(2) * sum(residuals .* S0 .* (term1 + term2));

    % --- df_trans ---
    df_trans = 2 * sum(residuals .* S0 .* (E_stick - E_ball)) * (f^2 * exp(x(3)));

    % --- dtheta ---
    dtheta = -2 * S0 * f * sum(residuals .* E_stick .* (-bvals * d .* 2 .* fibdotgrad) .* ...
        (cos(phi)*cos(theta) * qhat(:,1) + sin(phi)*cos(theta) * qhat(:,2) - sin(theta) * qhat(:,3)));

    % --- dphi ---
    dphi = -2 * S0 * f * sum(residuals .* E_stick .* (-bvals * d .* 2 .* fibdotgrad) .* ...
        (-sin(phi)*sin(theta) * qhat(:,1) + cos(phi)*sin(theta) * qhat(:,2)));

    grad = [dS0_trans, dd_trans, df_trans, dtheta, dphi];
end