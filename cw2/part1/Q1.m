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
% h: 假设检验结果(1=拒绝零假设，0=不拒绝零假设)
% p: p值，小于显著性水平(通常0.05)时拒绝零假设
% ci: 均值差异的95%置信区间
% stats: 包含t统计量(tstat)和自由度(df)的结构体
disp(['h value (reject the null hypothesis or not): ', num2str(h)]); % h=1 拒绝，h=0 不拒绝
disp(['p-value: ', num2str(p)]);
disp(['Confidence interval: ', num2str(ci')]);
disp(['t-statistic: ', num2str(stats.tstat)]);
disp(['Degrees of freedom: ', num2str(stats.df)]);
disp(['Combined standard deviation: ', num2str(stats.sd)]);