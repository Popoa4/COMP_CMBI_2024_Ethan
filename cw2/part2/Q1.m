% Single permutation test for the hypothesis that two groups have different means
%% (a) Simulate a similar dataset
n1 = 6;
n2 = 8;

student_id = 24046635;
rng(student_id);
group1_mean = 5;
group2_mean = 8;
group_std = 3;

group1_data = group1_mean + group_std * randn(n1,1);
group2_data = group2_mean + group_std * randn(n2,1);

[h, p, ci, stats] = ttest2(group1_data, group2_data);
% determine the t-statistic and p-value
disp(['t-statistic: ', num2str(stats.tstat)]);
disp(['p-value: ', num2str(p)]);

%% (b) compute the permutation-based p-value
% i. construct array D to store the observations
D = [group1_data; group2_data];
n = n1 + n2;

% ii. use nchoosek to construct all the permutations of D
% nchoosek(n, k) 返回从 n 个元素中选取 k 个元素的组合数。在这里，我们需要从 n 个元素（总样本量）中选取 n1 个元素（组 1 的样本量）的所有组合
perm_indices = nchoosek(1:n, n1);
n_perms = size(perm_indices, 1);
t_perms = zeros(n_perms, 1);

% iii. compute the t-statistic for all the permutations
for i = 1:n_perms
   g1 = D(perm_indices(i, :));
   g2 = D(setdiff(1:n, perm_indices(i, :)));
   mean_diff = mean(g1) - mean(g2);
   pooled_var = (var(g1)*(n1-1) + var(g2)*(n2-1)) / (n1+n2-2);
   t_perms(i) = mean_diff / sqrt(pooled_var*(1/n1 + 1/n2));
end


% iv. determine the p-value
% find the percentage of the permutations with t-statistic higher than origin
p_exact_t = sum(abs(t_perms) >= abs(stats.tstat)) / n_perms;
disp(['Exact permutation p-value (t-stat): ', num2str(p_exact_t)]);

%% (c) compute the p-value using the differences between the means
diff_perm = zeros(n_perms, 1);
diff_original = mean(group2_data) - mean(group1_data);
for i = 1:n_perms
    g1 = D(perm_indices(i, :));
    g2 = D(setdiff(1:n, perm_indices(i, :)));
    diff_perm(i) = mean(g1) - mean(g2);
end
p_exact_diff = sum(abs(diff_perm) >= abs(diff_original)) / n_perms;
disp(['Exact permutation p-value (mean diff): ', num2str(p_exact_diff)]);

%% (d) compute the approximate permutation-based p-value
% i. use randperm to estimate p-values
num_rand = 1000;
t_perm_rand = zeros(num_rand, 1);
indices_store = zeros(num_rand, n1); % 存放每次选中的组1的索引

% include the original label
original_idx = 1:n1;
t_perm_rand(1) = stats.tstat;
indices_store(1,:) = original_idx;

for i = 2:num_rand
    perm = randperm(n);
    % sort for the checking of the repeat
    idx1 = sort(perm(1:n1));
    idx2 = setdiff(1:n, idx1);
    sample1 = D(idx1);
    sample2 = D(idx2);
    t_perm_rand(i) = (mean(sample1)-mean(sample2)) / sqrt(var(sample1)/n1 + var(sample2)/n2);
    indices_store(i,:) = idx1;
end
% ii. compute the p-value
p_approx = sum(abs(t_perm_rand) >= abs(stats.tstat)) / num_rand;
disp(['Approximate permutation p-value: ', num2str(p_approx)]);

% iii. check if there are duplicates
[unique_indices, ia, ic] = unique(indices_store, 'rows');
num_unique = size(unique_indices, 1);
if num_unique < num_rand
    fprintf('%d duplicate permutations in 1000 permutations (%d effective permutations)\n', num_rand - num_unique, num_unique);
else
    fprintf('no duplicate in 1000 permutations.\n');
end
% 重复排列，会导致近似检验的有效重复数低于设定的1000
% 使得 p 值估计方差更大，不够准确。同时重复排列会使得总体抽样分布并非完全独立，进而影响统计推断。
