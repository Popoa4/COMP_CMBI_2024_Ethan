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

grad_dirs = A(1:3,:); % 3×N_measurements
qhat = grad_dirs';
G = A(4,:)'; 
delta = A(5,:)'; 
smalldel = A(6,:)'; 
TE = A(7,:)'; 
GAMMA = 2.675987E8; 

bvals = ((GAMMA * smalldel .* G).^2).*(delta - smalldel/3);
bvals = bvals/1e6; 

% Fit each model
[params_bs, resnorm_bs] = fit_ball_stick(meas, bvals, qhat);
[params_zs, resnorm_zs] = fit_zeppelin_stick(meas, bvals, qhat);
[params_zt, resnorm_zt] = fit_tortuosity(meas, bvals, qhat);

fprintf('===== Model comparison results =====\n');
fprintf('Ball and Stick RESNORM: %.4f\n', resnorm_bs);
fprintf('Zeppelin and Stick RESNORM: %.4f\n', resnorm_zs);
fprintf('Tortuosity Model RESNORM: %.4f\n', resnorm_zt);

%% Visualize fitting effect
figure;
subplot(3,1,1);
plot_model_fit(params_bs, 'Ball and Stick', meas, bvals, qhat);
subplot(3,1,2);
plot_model_fit(params_zs, 'Zeppelin and Stick', meas, bvals, qhat);
subplot(3,1,3);
plot_model_fit(params_zt, 'Tortuosity Model', meas, bvals, qhat);

function plot_model_fit(params, model_name, meas, bvals, grad_dirs)
    best_S0 = params(1);
    best_d = params(2);
    best_f = params(3);
    best_theta = params(4);
    best_phi = params(5);
    
    % Generate model prediction value
    fibdir = [cos(best_phi)*sin(best_theta), sin(best_phi)*sin(best_theta), cos(best_theta)];
    fibdotgrad = sum(grad_dirs .* fibdir, 2);
    S_pred = best_S0 * (best_f * exp(-bvals*best_d .* fibdotgrad.^2) + (1-best_f)*exp(-bvals*best_d));
    
    % Plot predictions vs actual signals
    scatter(bvals, meas, 5, 'b', 'filled'); hold on;
    scatter(bvals, S_pred, 5, 'r', 'filled');
    xlabel('b-value (s/mm²)');
    ylabel('Signal');
    legend('Measured', 'Predicted');
    title(model_name);
end