function nll = BallStickRicianNLL(x, Avox, bvals, qhat, snr_threshold)
    % 变换参数
    S0 = x(1)^2;
    d = x(2)^2;
    f = 1 / (1 + exp(-x(3)));
    theta = x(4);
    phi = x(5);
    sigma = x(6); % 获取sigma


    % 计算纤维方向
    fibdir = [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)];
    fibdotgrad = sum(qhat .* repmat(fibdir, [size(qhat,1), 1]), 2);
    
    % 计算模型信号
    S = S0 * (f * exp(-bvals*d .* (fibdotgrad.^2)) + (1-f) * exp(-bvals*d));
    snr = mean(S) / sigma;
    if snr > snr_threshold
        % 高SNR近似：使用SSD
        nll = sum((Avox - S).^2);
    else
        % 低SNR近似:瑞利分布
        nll = -sum(log(Avox) - 2*log(sigma) - Avox.^2./(2*sigma^2));

    end
     % 莱斯分布的负对数似然
    % nll = -sum(log(Avox) - 2*log(sigma) - (Avox.^2 + S.^2)./(2*sigma^2) + log(besseli(0, Avox.*S./sigma^2, 1))); % 包含修正贝塞尔函数, 第三个参数为1，表示对结果进行指数缩放。
end