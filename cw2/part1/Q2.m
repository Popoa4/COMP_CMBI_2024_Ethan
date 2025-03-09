%% (a) use ttest to compute the paired t-statistic
student_id = 24046635;
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

[h, p, ci, stats] = ttest2(group1_data, group2_data);
[h_paired, p_paired, ci_paired, stats_paired] = ttest(group1_data, group2_data);
% 配对t检验通常更有统计效力，因为它考虑了同一受试者在不同条件下的相关性
% 配对t检验的t值可能会比两样本t检验更大（绝对值），因为它消除了受试者间变异性
disp(['Previous two-sample t-statistic (stats.tstat): ', num2str(stats.tstat)]);
disp(['Paired t-statistic (stats_paired.tstat): ', num2str(stats_paired.tstat)]);

%% (b) compute the statistics with GLM model
% i. determine the matrix X and its rank
% constant variable
X0_paired = ones(n*2, 1); 
% explanatory variable for different time points
X1_paired = [ones(n,1); zeros(n,1)]; 
% the variable for the subjects
subject_indicators = zeros(n*2, n-1);
for i = 1:(n-1)
    subject_indicators([i, i+n], i) = 1;
end

X_paired = [X0_paired, X1_paired, subject_indicators];
rank_X_paired = rank(X_paired);
disp(['dim(X_paried) = ', num2str(rank_X_paired)]); 

% ii. determine the contrast vector
% Test the hypothesis that beta1 = 0
% whether there is a significant difference between the two time points
lambda_paired = zeros(size(X_paired, 2), 1);
% the position of time variable beta2
lambda_paired(2) = 1;

% iii. compute the t-statistic
beta_hat_paired = (X_paired' * X_paired) \ X_paired' * Y;

Y_hat_paired = X_paired * beta_hat_paired;
e_hat_paired = Y - Y_hat_paired;
sigma_hat_sq_paired = (e_hat_paired' * e_hat_paired) / (length(Y) - rank_X_paired);
S_beta_hat_paired = sigma_hat_sq_paired * inv(X_paired' * X_paired);

% compute the t-statistic by GLM
t_stat_paired_glm = (lambda_paired' * beta_hat_paired) / sqrt(lambda_paired' * S_beta_hat_paired * lambda_paired);

disp(['Paired t-statistic by GLM: ', num2str(t_stat_paired_glm)]);
disp(['Difference between them: ', num2str(t_stat_paired_glm - stats_paired.tstat)]);

