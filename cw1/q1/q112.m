clear; clc;

load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]);
load('bvecs');
qhat = bvecs';
bvals = 1000 * sum(qhat .* qhat, 2);
Avox = dwis(:,92,65,72);

% Define a starting point for the non-linear fit
startx = [3.5e+00, 3e-03, 2.5e-01, 0, 0];
% Define various options for the non-linear fitting algorithm.
h=optimset('MaxFunEvals',20000,...
 'Algorithm','quasi-newton',...
 'TolX',1e-10,...
 'TolFun',1e-10);
[parameter_hat, RESNORM, EXITFLAG, OUTPUT] = fminunc(@(x) BallStickSSD(x, Avox, bvals, qhat), startx, h);

format short e
disp('Fitted parameters (S0, d, f, theta, phi):');
disp(parameter_hat);
disp('RESNORM (SSD):');
disp(RESNORM);

%% visualization
S0 = parameter_hat(1);
d = parameter_hat(2);
f_val = parameter_hat(3);
theta = parameter_hat(4);
phi = parameter_hat(5);
plot_fit(phi, theta, S0, f_val, d, bvals, qhat, Avox, RESNORM);