function [S0_init, D, theta, phi] = dti_initialization(meas, bvals, grad_dirs)
    % DTI线性拟合
    % Y = [ones(size(bvals)), -bvals.*grad_dirs(:,1).^2, -2*bvals.*grad_dirs(:,1).*grad_dirs(:,2), ...
    %      -2*bvals.*grad_dirs(:,1).*grad_dirs(:,3), -bvals.*grad_dirs(:,2).^2, ...
    %      -2*bvals.*grad_dirs(:,2).*grad_dirs(:,3), -bvals.*grad_dirs(:,3).^2];
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
    
    % 主方向提
    [V, L] = eig(D);
    [~, idx] = max(diag(L));
    main_dir = V(:,idx);
    theta = acos(main_dir(3));
    phi = atan2(main_dir(2), main_dir(1));
    
    % % 输出参数
    % if nargout > 1
    %     varargout{1} = D;
    %     varargout{2} = theta;
    %     varargout{3} = phi;
    % else
    %     varargout{1} = theta;
    %     varargout{2} = phi;
    % end
end
