%% ---------------------- 1. 读取数据 ----------------------------
% 读取扩散信号数据（此处假设文件名与下载一致）
fid = fopen('isbi2015_data_normalised.txt', 'r', 'b');
fgetl(fid); % 跳过头部
D = fscanf(fid, '%f', [6, inf])';  % 得到数据矩阵，尺寸为 [3612 × 6]
fclose(fid);

% 选择第一个体素的测量数据
meas = D(:,1); % meas 为 3612×1 的信号向量

% 读取协议文件
fid = fopen('isbi2015_protocol.txt', 'r', 'b');
fgetl(fid);
A = fscanf(fid, '%f', [7, inf]);
fclose(fid);

% 构建协议变量
grad_dirs = A(1:3,:)';   % 将梯度方向转置为 [3612×3]，每行表示一个测量的梯度方向
G = A(4,:)';             % 梯度幅值
delta = A(5,:)';         % 梯度持续时间
smalldel = A(6,:)';      % 梯度上升时间
TE = A(7,:)';            % 回波时间 (这里可能用不上)
GAMMA = 2.675987E8;
% 计算 b 值 (单位为 s/m^2), 再转换为 s/mm^2
bvals = ((GAMMA * smalldel .* G).^2) .* (delta - smalldel/3);
bvals = bvals/1e6;   % bvals 现在单位为 s/mm^2

%% ---------------------- 2. 可视化信号在单位球上的分布 ----------------------------
figure;
scatter3(grad_dirs(:,1), grad_dirs(:,2), grad_dirs(:,3), 40, meas, 'filled');
xlabel('X'); ylabel('Y'); zlabel('Z');
title('单位球上各梯度方向对应的测量信号');
colorbar;
axis equal;
% 观察散点图，若颜色（信号）呈现多个局部高值区域，则提示可能有多纤维组

%% ---------------------- 3. 利用聚类判断是否存在多纤维组 ----------------------------
% 为了区分不同方向，我们采用 kmeans 聚类
% 这里我们使用加权的思想，即高信号的梯度点应更容易被分为不同群
% 先对信号进行归一化处理得到权重
w = (meas - min(meas))/(max(meas)-min(meas));  % 将测量值归一化到 [0,1]
% 为简单起见，我们直接对梯度方向做 kmeans 聚类
K = 2;  % 假设可能存在2个主要纤维方向
% 进行 kmeans 聚类（注意：这里对 grad_dirs 聚类，无权重版；更复杂的方法可考虑weighted kmeans）
[idx, C] = kmeans(grad_dirs, K, 'Replicates',10);

% 将每个聚类的平均信号计算出来
for k = 1:K
    cluster_signal_mean(k) = mean(meas(idx==k));
end
fprintf('聚类结果：\n');
for k = 1:K
    fprintf('聚类 %d 的平均信号: %.4f\n', k, cluster_signal_mean(k));
end
% 若两个聚类的平均信号均较高（例如相差不大且均明显高于背景），说明该体素可能包含两组纤维。

%% ---------------------- 4. 进一步的聚类诊断 ----------------------------
% 可以对聚类进行 silhouette 分析，检查最佳簇数
eva = evalclusters(grad_dirs, 'kmeans', 'silhouette', 'KList', 1:3);
figure;
plot(eva);
title('基于梯度方向的聚类评价 (Silhouette)');
xlabel('聚类数'); ylabel('轮廓系数');
% 如果最佳聚类数超过1，则表明数据内存在多个不同方向的信号峰结构。

%% ---------------------- 5. 输出结论 ----------------------------
if all(cluster_signal_mean > 0.8*mean(meas))
    disp('聚类结果显示该体素存在多个高信号区域，提示有可能存在多纤维组。');
else
    disp('聚类结果显示信号主要集中于某一方向，可能只有单一纤维组。');
end
