function [S0, d, f, theta, phi] = dti_initialization_single_voxel(Y, Avox)
    x_hat = Y \ log(Avox(:));
    
    D = [x_hat(2) x_hat(3) x_hat(4);
         x_hat(3) x_hat(5) x_hat(6);
         x_hat(4) x_hat(6) x_hat(7)];
    
    [V, L] = eig(D);
    lambda = diag(L);
    [~, idx] = max(abs(lambda));
    eig_vec = V(:,idx);
    MD = mean(lambda);
    FA = sqrt(3/2 * sum((lambda - MD).^2) / sum(lambda.^2));
    
    % Fiber direction
    theta = acos(eig_vec(3)); 
    phi = atan2(eig_vec(2), eig_vec(1)); 
    
    % Mapping to ball-and-stick model parameters
    S0 = exp(x_hat(1));
    d = MD;
    f = 1 - FA;
end
