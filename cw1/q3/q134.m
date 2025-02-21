%% 1. Load data and protocol
fid = fopen('isbi2015_data_normalised.txt', 'r', 'b');
fgetl(fid); 
D = fscanf(fid, '%f', [6, inf])'; 
fclose(fid);
meas = D(:,1); 
fid = fopen('isbi2015_protocol.txt', 'r', 'b');
fgetl(fid); 
A = fscanf(fid, '%f', [7, inf]);
fclose(fid);

grad_dirs = A(1:3,:); % 3×N_measurements
qhat = grad_dirs';
G = A(4,:)'; 
delta = A(5,:)'; 
smalldel = A(6,:)'; 
TE = A(7,:)'; 
GAMMA = 2.675987E8; 
bvals = ((GAMMA * smalldel .* G).^2).*(delta - smalldel/3);
bvals = bvals/1e6; 

[params_b2s, resnorm_b2s] = fit_ball_two_sticks(meas, bvals, qhat);
[params_b3s, resnorm_b3s] = fit_ball_multi_sticks(meas, bvals, qhat, 3);
[params_b5s, resnorm_b5s] = fit_ball_multi_sticks(meas, bvals, qhat, 5);
[params_b10s, resnorm_b10s] = fit_ball_multi_sticks(meas, bvals, qhat, 10);

% Results display
fprintf('===== Model comparison results =====\n');
fprintf('Ball and Two Stick RESNORM: %.4f\n', resnorm_b2s);
fprintf('Ball and Three Stick RESNORM: %.4f\n', resnorm_b3s);
fprintf('Ball and Five Stick RESNORM: %.4f\n', resnorm_b5s);
fprintf('Ball and Ten Stick RESNORM: %.4f\n', resnorm_b10s);
