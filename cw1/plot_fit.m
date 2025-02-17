function plot_fit(phi_fit, theta_fit, S0_fit, f_fit, d_fit, bvals, qhat, Avox, RESNORM_trans)
    % 计算模型信号
    fibdir = [cos(phi_fit) * sin(theta_fit), sin(phi_fit) * sin(theta_fit), cos(theta_fit)];
    fibdotgrad = sum(qhat .* repmat(fibdir, [size(qhat,1), 1]), 2);
    S_model = S0_fit * ( f_fit * exp(-bvals .* d_fit .* (fibdotgrad.^2)) + (1 - f_fit) * exp(-bvals .* d_fit) );
    
    % 生成测量编号 k（1 到 length(bvals)）
    k = (1:length(bvals))';
    
    % 绘图
    figure;
    hold on;
    % 绘制实际测量数据（蓝色点）
    plot(k, Avox, 'bo', 'MarkerFaceColor', 'b', 'DisplayName', 'Data');
    % 绘制模型预测数据（红色点）
    plot(k, S_model, 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Model');
    xlabel('k (Measurement Index)');
    ylabel('S (Signal Intensity)');
    title(sprintf('Ball-and-Stick Model Fit (Transformed)   SSD = %.4e', RESNORM_trans));
    legend('Location', 'best');
    
    % 在图形上添加 SSD 值
    dim = [0.75 0.6 0.2 0.2];
    str = sprintf('SSD = %.4e', RESNORM_trans);
    annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', 'BackgroundColor', 'w');
    
    hold off;
end
