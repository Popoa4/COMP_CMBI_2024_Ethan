clear;clc;
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat.*qhat, 2);

slice_num = 72;
[x_dim, y_dim] = size(dwis, 2:3);

% Initial parameters
startx_original = [3.5e3, 3e-3, 0.25, 0, 0];
startx_transformed(1) = sqrt(startx_original(1));
startx_transformed(2) = sqrt(startx_original(2));
startx_transformed(3) = -log((1/startx_original(3)) - 1);
startx_transformed(4) = startx_original(4);
startx_transformed(5) = startx_original(5);

h = optimset('MaxFunEvals', 20000, ...
             'Algorithm', 'quasi-newton', ...
             'TolX', 1e-10, ...
             'TolFun', 1e-10, ...
             'Display', 'off');
    

% 多次试验参数
num_trials = 3; 
perturb_scales = [0.2 * startx_transformed(1), ...
                  0.2 * startx_transformed(2), ...
                  0.2, 0.1 * pi, 0.2*pi];
tolerance = 1e-6; 

S0_map    = zeros(x_dim, y_dim);
d_map     = zeros(x_dim, y_dim);
f_map     = zeros(x_dim, y_dim);
RESNORM_map = zeros(x_dim, y_dim);
theta_map = zeros(x_dim, y_dim);
phi_map   = zeros(x_dim, y_dim); 

%% Traversing slices
for x = 1:x_dim
    for y = 1:y_dim
        voxel = squeeze(dwis(:, x, y, slice_num));
        if all(voxel > 0)
            Avox = voxel;
            RESNORM_values = zeros(1, num_trials);
            all_params = zeros(num_trials, 5);
            for i = 1:num_trials
                perturbation = randn(1, 5) .* perturb_scales;
                current_startx = startx_transformed + perturbation;
                current_startx(4) = mod(current_startx(4), pi); 
                current_startx(5) = mod(current_startx(5), 2*pi); 
                try
                    [param_hat_trans, RESNORM_trans, ~, ~] = fminunc(@(x) BallStickSSD_transformed(x, Avox, bvals, qhat), current_startx, h);
                    RESNORM_values(i) = RESNORM_trans;
                    all_params(i, :) = param_hat_trans;
                catch ME
                    fprintf('Trial %d has error: %s\n', i, ME.message);
                    % 1. Set RESNORM to a large value so that it will not be selected as the minimum in subsequent comparisons
                    RESNORM_values(i) = Inf;
                    % 2. You can also choose to set the parameters to initial values
                    all_params(i, :) = startx_transformed; 
                end
            end

            % Find the best fit
            is_minimal = abs(RESNORM_values - min(RESNORM_values)) <= tolerance * max(1, min(RESNORM_values));
            best_params_trans = all_params(find(is_minimal, 1), :);
            % Inverse transformation
            S0_best    = best_params_trans(1)^2;
            d_best     = best_params_trans(2)^2;
            f_best     = 1 / (1 + exp(-best_params_trans(3)));
            theta_best = best_params_trans(4);
            phi_best   = best_params_trans(5);
            S0_map(x, y)    = S0_best;
            d_map(x, y)     = d_best;
            f_map(x, y)     = f_best;
            RESNORM_map(x, y) = min(RESNORM_values);
            theta_map(x,y) = theta_best;
            phi_map(x,y) = phi_best;
        else
            S0_map(x, y)    = 0; 
            d_map(x, y)     = 0; 
            f_map(x, y)     = 0; 
            RESNORM_map(x, y) = NaN;
            theta_map(x,y) = NaN;
            phi_map(x,y) = NaN;

        end
    end
end

%% Show Parameter Mapping
figure;
subplot(2,2,1); imagesc(flipud(S0_map')); axis image off; colormap gray; title('S0 Map'); colorbar;
subplot(2,2,2); imagesc(flipud(d_map')); axis image off; colormap gray; title('d Map'); colorbar;
subplot(2,2,3); imagesc(flipud(f_map')); axis image off; colormap gray; title('f Map'); colorbar;
subplot(2,2,4); imagesc(flipud(RESNORM_map')); axis image off; colormap jet; title('RESNORM Map'); colorbar;

%% Fiber Orientation Map
% Calculate fiber orientation (and weight it with f)
[X, Y] = meshgrid(1:y_dim, 1:x_dim);
fibdir_x = f_map .* sin(theta_map) .* cos(phi_map);
fibdir_y = f_map .* sin(theta_map) .* sin(phi_map);

% Plot fiber orientation (quiver)
figure;
quiver(X, Y, flipud(fibdir_x), flipud(fibdir_y), 2.5);
title('Fiber Directions (weighted by f)');
axis image;
