%% (a) simulate sampled data from two groups
% set the seed of the sample group
% set the seed as my student id number
student_id = 123;
rng(student_id);

% set the size, means, variance
n = 20;

group1_mean = 1.5;
group1_std = 0.2;

group2_mean = 2.0;
group2_std = 0.2;

% sample the group data
group1_data = group1_mean + group1_std * randn(n,1);
group2_data = group2_mean + group2_std * randn(n,1);

% compute and print the mean and standard deviation of each sample
sample1_mean = mean(group1_data);
sample1_sd = std(group1_data);
sample2_mean = mean(group2_data);
sample2_sd = std(group2_data);

fprintf('Group1: mean = %.4f (expected: %.1f), std = %.4f (expected: %.1f)\n', ...
        sample1_mean, group1_mean, sample1_sd, group1_std);
fprintf('Group2: mean = %.4f (expected: %.1f), std = %.4f (expected: %.1f)\n', ...
        sample2_mean, group2_mean, sample2_sd, group2_std);

%% (b) compute the t-statistic of the two samples
[h, p, ci, stats] = ttest2(group1_data, group2_data);
% h: Hypothesis test result (1 = reject the null hypothesis, 0 = do not reject the null hypothesis)
% p: p value, reject the null hypothesis when it is less than the significance level (usually 0.05)
% ci: 95% confidence interval of the mean difference
% stats: Structure containing t statistic (tstat) and degrees of freedom (df)
% disp(['h value (reject the null hypothesis or not): ', num2str(h)]); % h=1 拒绝，h=0 不拒绝
% disp(['p-value: ', num2str(p)]);
% disp(['Confidence interval: ', num2str(ci')]);
disp(['t-statistic: ', num2str(stats.tstat)]);
disp(['Degrees of freedom: ', num2str(stats.df)]);
% disp(['Combined standard deviation: ', num2str(stats.sd)]);

%% (c) compute the t-statistic with GLM model
% -----------------------------------------------------------------------------
% i. derive the matrix X and its column space C(X)
X1 = [ones(n,1); zeros(n,1)];
X2 = [zeros(n,1); ones(n,1)];
X = [X1 X2];
% verify its space C(X)
dim_X = rank(X);
disp(['dim(X) = ', num2str(dim_X)]); 

% -----------------------------------------------------------------------------
% ii. derive the formula of the projection Px of C(x) and verify its properties
XtX = X' * X;
XtX_inv = inv(XtX);
disp('X''X:');
disp(XtX);
disp('(X''X)^(-1):');
disp(XtX_inv);

PX = X * XtX_inv * X';
% 对称性: PX' = PX
disp('Verify the symmetry of PX:');
disp(['|PX - PX''|=', num2str(norm(PX - PX'))]);
% 幂等性: PX × PX = PX
disp('Verify the idempotence of PX:');
disp(['|PX - PX*PX|=', num2str(norm(PX - PX*PX))]);
% 计算PX的迹
trace_PX = trace(PX);
disp(['trace of PX: ', num2str(trace_PX)]);
% PX的迹等于C(X)的维度，即dim(X)。在我们的例子中，trace(PX) = 2。这表示投影到C(X)后，保留了原始数据的2个自由度，对应于两个组的均值。

% -----------------------------------------------------------------------------
% iii. use PX to determine the projection of Y into C(X)
Y = [group1_data; group2_data];
% compute the projection
Y_hat = PX * Y;
% disp(['组1预测值: ', num2str(mean(Y_hat(1:20)))]);
% disp(['组2预测值: ', num2str(mean(Y_hat(21:40)))]);
% Ŷ是X列空间C(X)中与Y最接近的向量
% 在这个例子中，Ŷ前20个元素都等于第一组的样本均值，后20个元素都等于第二组的样本均值

% -----------------------------------------------------------------------------
% iv. compute Rx
I = eye(size(PX));
RX = I - PX;
% 对称性: RX' = RX
disp('Verify the symmetry of RX:');
disp(['|RX - RX''|=', num2str(norm(RX - RX'))]);
% 幂等性: RX × RX = RX
disp('Verify the idempotence of RX:');
disp(['|RX - RX*RX|=', num2str(norm(RX - RX*RX))]);

% -----------------------------------------------------------------------------
% v. use Rx to determine e_hat and the dimension of C(X)⊥
e_hat = RX * Y;
dim_CX_perp = size(X,1) - rank(X);
disp(['dimention of C(X): ', num2str(dim_CX_perp)]);

% -----------------------------------------------------------------------------
% vi. determine the angle between e_hat and Y_hat
angle_rad = acos(dot(e_hat, Y_hat) / (norm(e_hat) * norm(Y_hat)));
angle_deg = rad2deg(angle_rad);
disp(['the angle between e_hat and Y_hat is (degree): ', num2str(angle_deg)]);
% 因为 Y_hat 是 Y 在模型空间 C(X) 上的投影，而 e_hat 是 Y 在误差空间 C(X)⊥ 上的投影，这两个空间是正交的。夹角为 90 度。

% -----------------------------------------------------------------------------
% vii. use the formula to determine beta_hat
% β̂ = (X'X)⁻¹X'Y
beta_hat = XtX_inv * X' * Y;
% 叫最小二乘法估计的原因：使残差平方和 ||Y - Xβ||² 最小化

% -----------------------------------------------------------------------------
% viii. estimate the variance of the stochastic component
sigma_hat_sq = (e_hat' * e_hat) / (length(Y) - dim_X);
disp(['the estimated value of σ^2: ', num2str(sigma_hat_sq)]);
% 这就是MSE，因为分子(e_hat' * e_hat)是残差平方和(SSE)，分母(n - dim(X))是残差自由度

% -----------------------------------------------------------------------------
% ix. estimate the covariance matrix of the parameters and determine the STD of the parameters
S_beta_hat = sigma_hat_sq * XtX_inv;
disp('the covariance matrix of the parameters:');
disp(S_beta_hat);
std_beta_hat = sqrt(diag(S_beta_hat));
disp('STD of the parameters:');
disp(std_beta_hat);

% -----------------------------------------------------------------------------
% x. derive the contrast vector and reduced model
% H0: No difference between the means of the two groups
lambda = [1; -1];
mean_diff = lambda' * beta_hat;
disp(['difference between the estimated means(beta1-beta₂): ', num2str(mean_diff)]);
% 简化模型在群体均值相等的约束下，只需要一个表示共同均值的参数。因此，X₀是一个全1列向量
X0 = ones(length(Y), 1);

% -----------------------------------------------------------------------------
% xi. use X0to compute the additional error and F-statistic
PX0 = X0 * inv(X0'*X0) * X0'; 
Y_hat0 = PX0 * Y;
e_hat0 = Y - Y_hat0;
% compute the additional error
SSE_full = e_hat' * e_hat;
SSE_reduced = e_hat0' * e_hat0;
SSE_diff = SSE_reduced - SSE_full;
disp(['Additional error caused by constraints(SSE_diff): ', num2str(SSE_diff)]);

% Compute F-statistic and its DOF
% 分子自由度 = 全模型维度 - 简化模型维度
% 分母自由度 = 观测总数 - 全模型维度
df1 = rank(X) - rank(X0);
df2 = length(Y) - rank(X);
F_stat = (SSE_diff / df1) / (SSE_full / df2);
disp(['F-statistic: ', num2str(F_stat)]);
disp(['Degree of Fredom of F-statistic: ', num2str(df1), ', ', num2str(df2)]);

% -----------------------------------------------------------------------------
% xii. use t-statistic to determine which group has a higher mean
t_stat = (lambda' * beta_hat) / sqrt(lambda' * S_beta_hat * lambda);
disp(['t-statistic: ', num2str(t_stat)]);


% -----------------------------------------------------------------------------
% xiii. explain the meaning of the parameters
% compare with the value of ground truth
% beta1: represents the population mean of the first group
% beta2: represents the population mean of the second
true_beta = [group1_mean; group2_mean];
disp('estimated values, ground truth values, differences');
disp([beta_hat, true_beta, beta_hat - true_beta]);

% -----------------------------------------------------------------------------
% xiv. compute the projection of e
e_true_group1 = group1_data - group1_mean;
e_true_group2 = group2_data - group2_mean;
e_true = [e_true_group1; e_true_group2];
e_proj_CX = PX * e_true;
% 样本参数的偏差由投影到C(X)的误差分量决定
beta_bias = XtX_inv * X' * e_true;
disp('estimated beta difference:');
disp(beta_bias);
disp('real difference of beta:');
disp(beta_hat - true_beta);

% -----------------------------------------------------------------------------
% xv. compute the projection of e into C(X)⊥
e_proj_CX_perp = RX * e_true;
% 理论上，e_proj_CX_perp 应该等于 ê
% 因为RX * Y = RX * (Xβ + e) = RX * e，由于RX * X = 0
disp(['|e_proj_CX_perp - e_hat|: ', num2str(norm(e_proj_CX_perp - e_hat))]);

%% (d) compute the t-statistic with a new model (with X0, beta0)
X0_intercept = ones(2*n, 1);
X_d = [X0_intercept, X1, X2];
disp(['dim(X_d) = ', num2str(rank(X_d))]);

% ii. compute the new PX
PX_d = X_d * pinv(X_d'*X_d) * X_d'; 
disp(['|PX_d - PX| = ', num2str(norm(PX_d - PX))]);
% PX_d 和 PX 相同；估计空间相同：新模型只是对同一空间进行了不同的参数化；虽然X_d有3列，但它的列空间仍然是2维的，和之前的模型一样

% iii. determine the new contrast vector and reduced model
lambda_d = [0; 1; -1];
% 简化模型对应于H₀: λ'β = 0的约束，即只有截距项X₀β₀的模型，捕获整体均值。
X_d_reduced = X0_intercept;

% iv. compute the t-statistic
% compute the estimate of the parameter use pinv
beta_hat_d = pinv(X_d) * Y;
Y_hat_d = X_d * beta_hat_d;
e_hat_d = Y - Y_hat_d;
sigma_hat_sq_d = (e_hat_d' * e_hat_d) / (length(Y) - rank(X_d));
% Calculate the covariance matrix of parameter estimates
S_beta_hat_d = sigma_hat_sq_d * pinv(X_d' * X_d);
% compute the t-statistic
t_stat_d = (lambda_d' * beta_hat_d) / sqrt(lambda_d' * S_beta_hat_d * lambda_d);
disp(['new t-statistic with the constant variable: ', num2str(t_stat_d)]);
% β₀: 表示整体截距或基准水平
% β₁: 表示第一组的附加效应（相对于基准）
% β₂: 表示第二组的附加效应（相对于基准）
% 第一组均值 = β₀ + β₁
% 第二组均值 = β₀ + β₂

%% (e) compute the t-statistic with another model(without x2, beta2)
X_e = [X0_intercept, X1];
disp(['dim(X_e) = ', num2str(rank(X_e))]);

% ii. determine the new contrast vector and reduced model
lambda_e = [0; 1];
X_e_reduced = X0_intercept;

% iii. compute the t-statistic
beta_hat_e = pinv(X_e) * Y;
Y_hat_e = X_e * beta_hat_e;
e_hat_e = Y - Y_hat_e;
sigma_hat_sq_e = (e_hat_e' * e_hat_e) / (length(Y) - rank(X_e));
% Calculate the covariance matrix of parameter estimates
S_beta_hat_e = sigma_hat_sq_e * pinv(X_e' * X_e);
% compute the t-statistic
t_stat_e = (lambda_e' * beta_hat_e) / sqrt(lambda_e' * S_beta_hat_e * lambda_e);
disp(['new t-statistic dropping the X2: ', num2str(t_stat_e)]);
% β₀: 表示第二组的总体均值 (作为基准水平)
% β₁: 表示第一组相对于第二组的差异
