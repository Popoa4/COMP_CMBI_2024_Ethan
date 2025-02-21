function [S0_init, D, theta, phi] = dti_initialization(meas, bvals, grad_dirs)
    Y = zeros(length(bvals),7);
    for i = 1:length(bvals)
        b = bvals(i);
        q = grad_dirs(i,:);
        Y(i,:) = [1, -b*q(1)^2, -2*b*q(1)*q(2), -2*b*q(1)*q(3),...
                 -b*q(2)^2, -2*b*q(2)*q(3), -b*q(3)^2];
    end
    
    x = Y \ log(meas);
    S0_init = exp(x(1));
    D = [x(2) x(3) x(4); x(3) x(5) x(6); x(4) x(6) x(7)];
    
    % Main direction
    [V, L] = eig(D);
    [~, idx] = max(diag(L));
    main_dir = V(:,idx);
    theta = acos(main_dir(3));
    phi = atan2(main_dir(2), main_dir(1));
    
end
