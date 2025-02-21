clear; clc;

%% Data loading and preprocessing
load('data'); 
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]); 
load('bvecs'); 
qhat = bvecs';  
bvals = 1000 * sum(qhat.*qhat, 2); 

%% Select single voxel
Avox = dwis(:,92,65,72); 
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
             'Display', 'iter');

% Fit using the transformed objective function
[parameter_hat_trans, RESNORM_trans, EXITFLAG_trans, OUTPUT_trans] = fminunc(@(x) BallStickSSD_transformed(x, Avox, bvals, qhat), startx_transformed, h);
% Transform the fitting result back to the original parameter space
S0_fit    = parameter_hat_trans(1)^2;
d_fit     = parameter_hat_trans(2)^2;
f_fit     = 1 / (1 + exp(-parameter_hat_trans(3)));
theta_fit = parameter_hat_trans(4);
phi_fit   = parameter_hat_trans(5);

parameter_hat_original = [S0_fit, d_fit, f_fit, theta_fit, phi_fit];
disp('Fitted parameters (S0, d, f, theta, phi):');
disp(parameter_hat_original);
disp('RESNORM (transformed):');
disp(RESNORM_trans);

%% visualization
plot_fit(phi_fit, theta_fit, S0_fit, f_fit, d_fit, bvals, qhat, Avox, RESNORM_trans)