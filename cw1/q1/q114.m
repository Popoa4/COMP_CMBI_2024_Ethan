clear; clc;

%% Data loading and preprocessing
load('data');         
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);  
load('bvecs');        
qhat = bvecs';        
bvals = 1000 * sum(qhat.*qhat, 2);  
voxel_coords = [92, 65, 72]; % 用于测试的体素坐标，之后可以更改
Avox = dwis(:, voxel_coords(1), voxel_coords(2), voxel_coords(3));

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
%% Multiple fitting
num_trials = 150;
RESNORM_values = zeros(1, num_trials);
all_params = zeros(num_trials, 5);

% Define the perturbation scale
perturb_scales = [0.2 * startx_transformed(1), ...  
                   0.2 * startx_transformed(2), ...  
                   0.1* startx_transformed(3), ...           
                   0.1 * startx_transformed(4), ...           
                   0.2* startx_transformed(5)];            


for i = 1:num_trials
    % Generate random disturbances (normal distribution)
    perturbation = randn(1, 5) .* perturb_scales;
    current_startx = startx_transformed + perturbation;
    % Modulate theta and phi to ensure they are within the valid range
    current_startx(4) = mod(current_startx(4), pi); 
    current_startx(5) = mod(current_startx(5), 2*pi); 

    [param_hat_trans, RESNORM_trans, ~, ~] = fminunc(@(x) BallStickSSD_transformed(x, Avox, bvals, qhat), current_startx, h);
    RESNORM_values(i) = RESNORM_trans;
    all_params(i, :) = param_hat_trans;
end

%% Analyze the results
[min_RESNORM, min_index] = min(RESNORM_values);
best_params_trans = all_params(min_index, :);

% Transform the optimal parameters back to the original space
S0_best    = best_params_trans(1)^2;
d_best     = best_params_trans(2)^2;
f_best     = 1 / (1 + exp(-best_params_trans(3)));
theta_best = best_params_trans(4);
phi_best   = best_params_trans(5);

best_params_original = [S0_best, d_best, f_best, theta_best, phi_best];
plot_fit(phi_best, theta_best, S0_best, f_best, d_best, bvals, qhat, Avox, min_RESNORM);

fprintf('Minimum RESNORM value: %.4e\n', min_RESNORM);
fprintf('Corresponding original space parameters(S0, d, f, theta, phi):\n');
disp(best_params_original);

tolerance = 1e-4;

proportion_best = sum(abs(RESNORM_values - min_RESNORM) <= tolerance) / num_trials;
disp(sum(abs(RESNORM_values - min_RESNORM) < tolerance));
fprintf('Find the proportion of trials close to the minimum RESNORM value: %.2f\n', proportion_best);


% Estimate the number of trials required to achieve 95% confidence
if proportion_best > 0 && proportion_best < 1 
    n_95 = ceil(log(0.05) / log(1 - proportion_best));
    fprintf('The number of trials required to reach 95%% confidence: %d\n', n_95);
else
    fprintf('Unable to estimate the number of trials required to reach 95% confidence (proportion is 0 or 1).\n');
end

%% Multi-voxel validation
test_voxels = [93,65,72; 
               80,50,70; 
               100,80,72]; 
for v = 1:size(test_voxels,1)
    current_voxel = test_voxels(v,:);
    Avox = dwis(:,current_voxel(1),current_voxel(2),current_voxel(3));
    
    num_trials = 100;
    local_RESNORM = zeros(1,num_trials);
    
    parfor i = 1:num_trials
        perturbation = randn(1,5).*perturb_scales;
        [~, RESNORM] = fminunc(@(x)BallStickSSD_transformed(x,Avox,bvals,qhat),...
                              startx_transformed + perturbation, h);
        local_RESNORM(i) = RESNORM;
    end
    
    min_local = min(local_RESNORM);
    success_rate = sum(abs(local_RESNORM - min_local) <= tolerance) / num_trials;
    
    fprintf('\n Voxel [%d,%d,%d]:\n', current_voxel);
    fprintf('Minimum RESNORM: %.3e Success rate: %.2f\n', min_local, success_rate);
end
