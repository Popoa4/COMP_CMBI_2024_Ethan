% load the mask
data_folder = './data';
mask_fid = fopen(fullfile(data_folder, 'wm_mask.img'), 'r', 'l');
% if mask_fid ==-1
%     error('文件打开失败！请检查路径和权限');
% end
mask = fread(mask_fid, 'float');
mask = reshape(mask, [40, 40, 40]);
fclose(mask_fid);

roi_indices = find(mask > 0);
num_voxels_roi = length(roi_indices);
% load the FA map
n1 = 8;
n2 = 8;
n = n1 + n2;
all_data = zeros(num_voxels_roi, n);
% disp(size(all_data(:,1)));
% load the data of group 1
n1 = 8;
id = 1;
for i = 4:11
    % fid = fopen(group2_files(i).name, 'r', 'l');
    filename = sprintf('./data/CPA%d_diffeo_fa.img', i);
    % disp(filename);
    fid = fopen(filename, 'r', 'l');
    data = fread(fid, 'float');
    data = reshape(data, [40 40 40]);
    fclose(fid);
    
    % store the data within ROI
    all_data(:, id) = data(roi_indices);
    id = id +1;
end

% load the data of group 2
n2 = 8;
IDs = [3,6,9,10,13,14,15,16];
id=1;
for i = IDs
    % fid = fopen(group2_files(i).name, 'r', 'l');
    filename = sprintf('./data/PPA%d_diffeo_fa.img', i);
    % disp(filename);
    fid = fopen(filename, 'r', 'l');
    data = fread(fid, 'float');
    data = reshape(data, [40 40 40]);
    fclose(fid);
    all_data(:, id+n1) = data(roi_indices);
    id = id +1;
end
%% (a) compute the two-sample t-statistic between the groups
X = [ones(n, 1), [zeros(n1, 1); ones(n2, 1)]];
t_stats_original = zeros(num_voxels_roi, 1);
% compute t-statistic for every voxel using GLM
for i = 1:num_voxels_roi
    Y = all_data(i, :)';  % 当前体素的数据
    % disp(['X 的维度: ', mat2str(size(X))])
    % disp(['Y 的维度: ', mat2str(size(Y))])
    b = X \ Y;
    residuals = Y - X * b;
    sigma2 = sum(residuals.^2) / (n - 2); 
    var_b = sigma2 * inv(X' * X);
    t_stats_original(i) = b(2) / sqrt(var_b(2, 2));
end

max_t_original = max(abs(t_stats_original));
fprintf('The maximum t statistic under the original label: %.4f\n', max_t_original);

%% (b) use the 1(b) strategy to determine all the permutation of group labels
% perm_indices = nchoosek(1:n, n1);
% n_perms = size(perm_indices, 1);
% disp(n_perms);
% % store the max t-statistic for every permutation
% max_t_perms = zeros(n_perms, 1);


n_random_perms = 1000; % 设定随机排列次数（包括原始排列）
max_t_perms = zeros(n_random_perms, 1);

% 总是包含原始数据的结果
max_t_perms(1) = max_t_original;
parfor p = 2:n_random_perms
    rand_indices = randperm(n);
    % 根据随机索引创建新的组标签
    perm_labels = [zeros(n1, 1); ones(n2, 1)]; % 原始标签
    perm_labels = perm_labels(rand_indices);   % 打乱标签顺序
    % 新的设计矩阵
    X_perm = [ones(n, 1), perm_labels];
    % 重新计算每个体素的 t 统计量
    t_stats_perm = zeros(num_voxels_roi, 1);
    for v = 1:num_voxels_roi
        Y = all_data(v, :)';
        b = X_perm \ Y;
        residuals = Y - X_perm * b;
        sigma2 = sum(residuals.^2) / (n - 2);
        var_b = sigma2 * inv(X_perm' * X_perm);
        t_stats_perm(v) = b(2) / sqrt(var_b(2, 2));
    end
    % 记录当前排列的最大 t 统计量
    max_t_perms(p) = max(abs(t_stats_perm));
end
%% (c) computer the p-value corrected
p_corrected = sum(max_t_perms >= max_t_original) / n_perms;
fprintf('Corrected p-value: %.4f\n', p_corrected);

%% (d) determine the max t-statistic threshold corresponding to p-value of 5%
% sort the max_t_perms
sorted_max_t = sort(max_t_perms, 'descend');

% find the threshold of 0.05
threshold_5pct = sorted_max_t(ceil(0.05 * n_random_perms));

fprintf('maximum t statistic threshold corresponding to the 5%% significance level: %.4f\n', threshold_5pct);
