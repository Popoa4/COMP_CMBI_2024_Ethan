function [S0_map, d_map, f_map, theta_map, phi_map] = dti_initialization(dwis, bvals, qhat, slice_num)
    [x_dim, y_dim] = size(dwis, 2:3);
    Y = build_design_matrix(bvals, qhat);
    [S0_map, MD_map, FA_map, eig_vec_map] = deal(zeros(x_dim, y_dim));
    
    for x = 1:x_dim
        for y = 1:y_dim
            Avox = squeeze(dwis(:,x,y,slice_num));
            if all(Avox>0)
                x_hat = Y \ log(Avox(:));
                D = [x_hat(2) x_hat(3) x_hat(4);...
                     x_hat(3) x_hat(5) x_hat(6);...
                     x_hat(4) x_hat(6) x_hat(7)];
                [V, L] = eig(D);
                lambda = diag(L);
                MD = mean(lambda);
                FA = sqrt(3/2 * sum((lambda - MD).^2) / sum(lambda.^2));
                [~, idx] = max(abs(lambda));
                eig_vec = V(:,idx);
                
                S0_map(x,y) = exp(x_hat(1));
                MD_map(x,y) = MD;
                FA_map(x,y) = FA;
                theta_map(x,y) = acos(eig_vec(3));
                phi_map(x,y) = atan2(eig_vec(2), eig_vec(1));
            end
        end
    end
    d_map = MD_map;
    f_map = FA_map; % 可根据需要调整映射关系
end
