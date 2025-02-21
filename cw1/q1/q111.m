%% Data loading and preprocessing
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);

%% Construct design matrix Y
Y = build_design_matrix(bvals, qhat);


%% Single voxel test
voxel = dwis(:, 92, 65, 72);
A = log(voxel(:));
x = Y \ A;
% Build diffusion tensor
D = [x(2) x(3) x(4);
     x(3) x(5) x(6);
     x(4) x(6) x(7)];

MD = (x(2) + x(5) + x(7)) / 3;
[V, L] = eig(D);
lambda = diag(L);
lambda_mean = mean(lambda);
FA = sqrt( (3/2) * sum((lambda - lambda_mean).^2) / sum(lambda.^2) );
disp("MD:");
disp(MD);
disp("FA");
disp(FA);

%% Full slice processing (slice 72)
slice_num = 72;
[x_dim, y_dim] = size(dwis, 2:3);

MD_map = zeros(x_dim, y_dim);
FA_map = zeros(x_dim, y_dim);
color_map = zeros(x_dim, y_dim, 3);

for x = 1:x_dim
    for y = 1:y_dim
        voxel = squeeze(dwis(:, x, y, slice_num));
        if all(voxel > 0)
            A = log(voxel(:));
            x_hat = Y \ A;
            D = [x_hat(2) x_hat(3) x_hat(4);
                 x_hat(3) x_hat(5) x_hat(6);
                 x_hat(4) x_hat(6) x_hat(7)];
            
            MD_map(x,y) = (x_hat(2) + x_hat(5) + x_hat(7)) / 3;
            [V, L] = eig(D);
            lambda = diag(L);
            [~, idx] = max(abs(lambda));
            eig_vec = V(:,idx);
            FA_val = sqrt( (3/2) * sum((lambda - mean(lambda)).^2) / sum(lambda.^2) );
            FA_map(x,y) = FA_val;
            color_map(x,y,:) = abs(eig_vec') * FA_val;
        else
            MD_map(x,y) = 0;
            FA_map(x,y) = 0;
            color_map(x,y,:) = 0;
        end
    end
end

%% visualization
figure;

subplot(1,3,1);
imagesc(flipud(MD_map'));
axis image off; 
colormap gray; 
title('Mean Diffusivity');

subplot(1,3,2);
imagesc(flipud(FA_map')); 
axis image off; 
colormap gray; 
title('Fractional Anisotropy');

subplot(1,3,3);
color_map_normalized = color_map / max(color_map(:));
image(flipud(permute(color_map_normalized, [2,1,3])));
axis image off; 
title('Direction Encoded Color');
