clear; clc; close all;
%% 1. Loading data and protocols
fid = fopen('isbi2015_data_normalised.txt', 'r', 'b');
fgetl(fid); 
D = fscanf(fid, '%f', [6, inf])'; 
fclose(fid);
meas = D(:,1); 
fid = fopen('isbi2015_protocol.txt', 'r', 'b');
fgetl(fid);
A = fscanf(fid, '%f', [7, inf]);
fclose(fid);

% Creating a protocol variable
grad_dirs = A(1:3,:); % 3×N_measurements
qhat = grad_dirs';
G = A(4,:)';
delta = A(5,:)'; 
smalldel = A(6,:)'; 
TE = A(7,:)'; 
GAMMA = 2.675987E8;

% b s/m^2), transform to s/mm^2
bvals = ((GAMMA * smalldel .* G).^2).*(delta - smalldel/3);
bvals = bvals/1e6; % s/mm^2

%% Model Fitting Settings
sigma_noise = 0.04;
N = length(meas);


%% DTI-based initialization
Y = [ones(length(bvals),1), -bvals.*(grad_dirs(1,:)'.^2), -2*bvals.*grad_dirs(1,:)'.*grad_dirs(2,:)', ...
     -2*bvals.*grad_dirs(1,:)'.*grad_dirs(3,:)', -bvals.*(grad_dirs(2,:)'.^2), ...
     -2*bvals.*grad_dirs(2,:)'.*grad_dirs(3,:)', -bvals.*(grad_dirs(3,:)'.^2)];

x_dti = Y \ log(meas);
S0_init = exp(x_dti(1));
D = [x_dti(2) x_dti(3) x_dti(4); 
     x_dti(3) x_dti(5) x_dti(6); 
     x_dti(4) x_dti(6) x_dti(7)];

[V, L] = eig(D);
[~, idx] = max(diag(L));
main_dir = V(:,idx);
theta_init = acos(main_dir(3));
phi_init = atan2(main_dir(2), main_dir(1));

startx_original = [S0_init, trace(D)/3, 0.5, theta_init, phi_init]; % d=MD, f=0.5


options = optimoptions('fmincon',...
    'Algorithm', 'sqp',...       
    'MaxFunctionEvaluations', 1e4,...
    'StepTolerance', 1e-8,...
    'OptimalityTolerance', 1e-6,...
    'Display', 'iter');

lb = [0,   0,   0,   0,    0];   
ub = [Inf, Inf, 1,   pi, 2*pi];  


%% Multiple fitting attempts, collecting RESNORM
num_trials = 100; 
RESNORM_values = zeros(num_trials,1);
params_all = zeros(num_trials,5);

perturb_scales = [0.5 * startx_original(1), ...  
                   0.5 * startx_original(2), ...  
                   0.5* startx_original(3), ...                   
                   0.5 * startx_original(4), ...              
                   0.5* startx_original(5)];                
for i = 1:num_trials
    perturbation = randn(1,5).*perturb_scales;
    current_startx = startx_original + perturbation;
    try
        [params_hat_trans, RESNORM] = fmincon(@(x)BallStickSSD_Enhanced(x,meas,bvals,qhat),...
                          current_startx, [], [], [], [], lb, ub, @nonlcon, options);
        
    catch ME
        fprintf('Trial %d error: %s\n', i, ME.message);
        RESNORM = Inf;
        params_hat_trans = current_startx;
    end
    RESNORM_values(i) = RESNORM;
    params_all(i,:) = params_hat_trans;
end

tol_for_global = 1e-8;
min_RESNORM = min(RESNORM_values);
global_trials = sum(abs(RESNORM_values - min_RESNORM) <= tol_for_global * max(1, min_RESNORM));
fprintf('Minimum RESNORM = %.4e\n', min_RESNORM);
fprintf('%d of %d trials found the same global minimum (ratio = %.2f%%)\n', num_trials, global_trials, 100*global_trials/num_trials);

%% Get the best fit parameters and transform back to the original parameters
[min_RESNORM, min_index] = min(RESNORM_values);
best_params_trans = params_all(min_index, :);
disp(best_params_trans);

fprintf('Best fit parameters (original space）:\n');
fprintf('S0 = %.2f\n', best_params_trans(1));
fprintf('d = %.6f\n', best_params_trans(2));
fprintf('f = %.4f\n', best_params_trans(3));
fprintf('\nRESNORM is expected to be ≈ %f (36120.04^2 ≈ 5.78), the minimum RESNORM = %.4e\n', N*sigma_noise^2, min_RESNORM);



%% 可视化拟合效果
best_S0 = best_params_trans(1);
best_d = best_params_trans(2);
best_f = best_params_trans(3);
best_theta = best_params_trans(4);
best_phi = best_params_trans(5);

% 生成模型预测值
fibdir = [cos(best_phi)*sin(best_theta), sin(best_phi)*sin(best_theta), cos(best_theta)];
fibdotgrad = sum(grad_dirs' .* fibdir, 2);
S_pred = best_S0 * (best_f * exp(-bvals*best_d .* fibdotgrad.^2) + (1-best_f)*exp(-bvals*best_d));

% 绘制预测 vs 实际信号
figure;
scatter(bvals, meas, 5, 'b', 'filled'); hold on;
scatter(bvals, S_pred, 5, 'r', 'filled');
xlabel('b-value (s/mm²)');
ylabel('Signal');
legend('Measured', 'Predicted');
title('Ball-and-Stick Model Fit Quality');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [c, ceq] = nonlcon(x)
    % 非线性不等式约束 c(x) ≤ 0
    c = []; 
    % 非线性等式约束 ceq(x) = 0
    ceq = [];
    % 示例：强制纤维方向与Z轴夹角小于60度
    % theta = x(4);
    % c = [theta - pi/3]; % theta ≤ π/3
end