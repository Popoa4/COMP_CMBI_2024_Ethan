function [S0, d, f, theta, phi] = dti_initialization_single_voxel(Y, Avox)
    % 线性拟合
    x_hat = Y \ log(Avox(:));
    
    % 构建扩散张量
    D = [x_hat(2) x_hat(3) x_hat(4);
         x_hat(3) x_hat(5) x_hat(6);
         x_hat(4) x_hat(6) x_hat(7)];
    
    % 计算特征值和特征向量
    [V, L] = eig(D);
    lambda = diag(L);
    [~, idx] = max(abs(lambda));
    eig_vec = V(:,idx);
    
    % 计算平均扩散率（MD）和各向异性分数（FA）
    MD = mean(lambda);
    FA = sqrt(3/2 * sum((lambda - MD).^2) / sum(lambda.^2));
    
    % 纤维方向
    theta = acos(eig_vec(3)); % 与z轴的夹角
    phi = atan2(eig_vec(2), eig_vec(1)); % 在xy平面内的角度
    
    % 映射到球棍模型参数
    S0 = exp(x_hat(1));
    d = MD;
    f = 1 - FA; % 可根据实际情况调整
end
