%% 数据加载与预处理
load('data');
dwis = double(dwis);
dwis = permute(dwis, [4,1,2,3]); % 调整维度为[108,145,174,145]

% 加载梯度方向并计算b值
load('bvecs'); % 假设bvecs是3x108的矩阵
qhat = bvecs'; % 转置为108x3
bvals = 1000 * sum(qhat .* qhat, 2); % 计算b值

%% 构建设计矩阵Y
Y = build_design_matrix(bvals, qhat);
% Y = zeros(108, 7);
% for i = 1:108
%     b = bvals(i);
%     q = qhat(i, :);
%     % q(1)->q_x; q(2)->q_y; q(3)->q_z; 
%     Y(i,:) = [1, -b*q(1)^2, -2*b*q(1)*q(2), -2*b*q(1)*q(3), -b*q(2)^2, -2*b*q(2)*q(3), -b*q(3)^2];
% end

%% 单个体素测试
voxel = dwis(:, 92, 65, 72);
A = log(voxel(:)); % 确保列向量
x = Y \ A; % 线性回归求解

% 构建扩散张量
D = [x(2) x(3) x(4);
     x(3) x(5) x(6);
     x(4) x(6) x(7)];

% 计算平均扩散率(MD)
MD = (x(2) + x(5) + x(7)) / 3;

% 计算各向异性分数(FA)
[V, L] = eig(D);
lambda = diag(L);
lambda_mean = mean(lambda);
FA = sqrt( (3/2) * sum((lambda - lambda_mean).^2) / sum(lambda.^2) );
disp("MD:");
disp(MD);
disp("FA");
disp(FA);

%% 全切片处理（切片72）
slice_num = 72;
[x_dim, y_dim] = size(dwis, 2:3); % 145x174

% 预分配内存
MD_map = zeros(x_dim, y_dim);
FA_map = zeros(x_dim, y_dim);
color_map = zeros(x_dim, y_dim, 3); % RGB颜色映射

for x = 1:x_dim
    for y = 1:y_dim
        % 提取信号并处理无效值
        voxel = squeeze(dwis(:, x, y, slice_num));
        if all(voxel > 0)
            A = log(voxel(:));
            x_hat = Y \ A;
            
            % 构建扩散张量
            D = [x_hat(2) x_hat(3) x_hat(4);
                 x_hat(3) x_hat(5) x_hat(6);
                 x_hat(4) x_hat(6) x_hat(7)];
            
            % 计算MD
            MD_map(x,y) = (x_hat(2) + x_hat(5) + x_hat(7)) / 3;
            
            % 计算FA和主方向
            [V, L] = eig(D);
            lambda = diag(L);
            [~, idx] = max(abs(lambda));
            eig_vec = V(:,idx);
            
            FA_val = sqrt( (3/2) * sum((lambda - mean(lambda)).^2) / sum(lambda.^2) );
            FA_map(x,y) = FA_val;
            
            % 方向编码颜色（RGB对应XYZ方向）
            color_map(x,y,:) = abs(eig_vec') * FA_val;
        else
            MD_map(x,y) = 0;
            FA_map(x,y) = 0;
            color_map(x,y,:) = 0;
        end
    end
end

%% 结果可视化
figure;

% 平均扩散率图
subplot(1,3,1);
imagesc(flipud(MD_map')); % 转置并翻转以正确显示
axis image off; 
colormap gray; 
title('Mean Diffusivity');

% 各向异性分数图
subplot(1,3,2);
imagesc(flipud(FA_map')); 
axis image off; 
colormap gray; 
title('Fractional Anisotropy');

% 方向编码颜色图
subplot(1,3,3);
color_map_normalized = color_map / max(color_map(:));
image(flipud(permute(color_map_normalized, [2,1,3])));
axis image off; 
title('Direction Encoded Color');

%% 辅助函数（可选）
% 可添加额外的函数用于数据清洗或可视化增强
