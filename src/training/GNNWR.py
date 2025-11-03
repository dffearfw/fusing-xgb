import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


class SpatialWeightCalculator:
    """空间权重计算器"""

    def __init__(self, bandwidth=None, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def calculate_weights(self, coords_train, coords_predict, bandwidth=None):
        """计算空间权重矩阵"""
        if bandwidth is None:
            bandwidth = self.bandwidth

        # 计算地理距离
        distances = cdist(coords_train, coords_predict, metric='euclidean')

        # 根据核函数计算权重
        if self.kernel == 'gaussian':
            weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        elif self.kernel == 'exponential':
            weights = np.exp(-distances / bandwidth)
        elif self.kernel == 'bisquare':
            weights = np.where(distances <= bandwidth,
                               (1 - (distances / bandwidth) ** 2) ** 2, 0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        return weights

    def adaptive_bandwidth(self, coords, k=10):
        """自适应带宽 - 使用k近邻距离"""
        distances = cdist(coords, coords, metric='euclidean')
        # 对每个点，取第k个最近邻的距离作为带宽
        sorted_distances = np.sort(distances, axis=1)
        bandwidths = sorted_distances[:, k]
        return bandwidths.mean()  # 返回平均带宽


class EnhancedGNNWRModel(nn.Module):
    """增强的地理神经网络加权回归模型"""

    def __init__(self, input_dim, hidden_dims=[64, 32, 16], output_dim=1,
                 spatial_dim=2, use_attention=True):
        super(EnhancedGNNWRModel, self).__init__()
        self.use_attention = use_attention

        # 主特征处理网络
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.feature_network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # 空间注意力机制
        if use_attention:
            self.spatial_attention = nn.Sequential(
                nn.Linear(spatial_dim + input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, x, spatial_weights=None, coords=None):
        # 特征提取
        features = self.feature_network(x)  # 形状: [batch_size, hidden_dim]

        # 空间注意力加权 - 改进版本
        if self.use_attention and spatial_weights is not None and coords is not None:
            # 方法：空间平滑 - 使用邻近样本的特征进行加权平均
            batch_size = features.shape[0]
            hidden_dim = features.shape[1]

            # 归一化空间权重
            row_sums = torch.sum(spatial_weights, dim=1, keepdim=True)
            normalized_weights = spatial_weights / torch.where(row_sums > 0, row_sums, torch.tensor(1.0))

            # 应用空间平滑：weighted_features[i] = Σ_j normalized_weights[i,j] * features[j]
            weighted_features = torch.matmul(normalized_weights, features)

            output = self.output_layer(weighted_features)
        else:
            # 没有空间权重
            output = self.output_layer(features)

        return output.squeeze()


class SpatialAutocorrelation:
    """空间自相关性分析"""

    @staticmethod
    def morans_i(values, weight_matrix):
        """计算莫兰指数"""
        n = len(values)
        values = np.array(values)
        mean_val = values.mean()

        numerator = 0
        denominator = 0

        for i in range(n):
            for j in range(n):
                numerator += weight_matrix[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val) ** 2

        morans_i = (n / np.sum(weight_matrix)) * (numerator / denominator)
        return morans_i

    @staticmethod
    def calculate_spatial_lag(values, weight_matrix):
        """计算空间滞后项"""
        return np.dot(weight_matrix, values)


class EnhancedSpatialDataset(Dataset):
    """增强的空间数据集"""

    def __init__(self, features, targets, coords, spatial_weights=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.coords = torch.FloatTensor(coords)
        self.spatial_weights = torch.FloatTensor(spatial_weights) if spatial_weights is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.spatial_weights is not None:
            return self.features[idx], self.targets[idx], self.coords[idx], self.spatial_weights[idx]
        else:
            return self.features[idx], self.targets[idx], self.coords[idx]


class EnhancedGNNWRTrainer:
    """增强的GNNWR训练器 - 内存优化版本"""

    def __init__(self, input_dim, coords, hidden_dims=[64, 32, 16],
                 learning_rate=0.001, bandwidth=None, use_spatial_weights=True,
                 max_samples_for_spatial=5000, device='auto'):

        # 添加 logger 初始化
        self.logger = logging.getLogger("EnhancedGNNWRTrainer")

        # 自动检测或指定设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.logger.info(f"使用设备: {self.device}")

        if self.device.type == 'cuda':
            self.logger.info(f"GPU信息: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 初始化模型并移动到设备
        self.model = EnhancedGNNWRModel(input_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()



        print(f"=== EnhancedGNNWRTrainer 初始化调试 ===")
        print(f"输入 coords 类型: {type(coords)}")
        print(f"输入 coords is None: {coords is None}")
        print(f"输入 coords id: {id(coords)}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if coords is not None:
            self.coords = coords.copy()  # 创建副本，避免修改原数据
        else:
            self.coords = None

        self.use_spatial_weights = use_spatial_weights

        print(f"EnhancedGNNWRTrainer 初始化完成")
        print(f"保存的 coords id: {id(self.coords)}")

        # # 检查样本数量，如果太多则禁用空间权重
        # if coords is not None and len(coords) > max_samples_for_spatial:
        #     self.logger.warning(f"样本数量 {len(coords)} 超过限制 {max_samples_for_spatial}，禁用空间权重")
        #     self.use_spatial_weights = False

        # 初始化空间权重计算器（仅在需要时）
        if self.use_spatial_weights and coords is not None:
            self.weight_calculator = SpatialWeightCalculator(bandwidth=bandwidth)
            if bandwidth is None:
                # 使用较小的k值以减少计算量
                bandwidth = self.weight_calculator.adaptive_bandwidth(coords, k=5)
                self.weight_calculator.bandwidth = bandwidth

            # 计算全局空间权重矩阵（分批计算以避免内存问题）
            self.global_weights = self._compute_sparse_weights(coords)
        else:
            self.weight_calculator = None
            self.global_weights = None

        # 初始化模型
        self.model = EnhancedGNNWRModel(input_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        self.logger.info(f"EnhancedGNNWR训练器初始化完成，使用空间权重: {self.use_spatial_weights}")

    def _compute_gpu_spatial_weights(self, batch_coords):
        """在GPU上计算空间权重"""
        n_batch = batch_coords.shape[0]

        if n_batch <= 1:
            return torch.ones((n_batch, n_batch), device=self.device)

        # 使用PyTorch向量化计算（GPU加速）
        diff = batch_coords.unsqueeze(1) - batch_coords.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=2))

        # 自适应带宽
        sorted_distances = torch.sort(distances, dim=1).values
        bandwidth = torch.mean(sorted_distances[:, 1:6])  # 前5个邻居

        # 计算权重
        weights = torch.exp(-0.5 * (distances / bandwidth) ** 2)

        # 归一化
        row_sums = torch.sum(weights, dim=1, keepdim=True)
        weights = weights / torch.where(row_sums > 0, row_sums, torch.tensor(1.0))

        return weights

    def create_optimized_dataloader(dataset, batch_size=32, num_workers=None, pin_memory=True):
        """创建优化的数据加载器"""
        if num_workers is None:
            # 自动设置工作进程数
            num_workers = min(8, os.cpu_count() // 2)  # 使用一半的CPU核心

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,  # 加速GPU数据传输
            persistent_workers=True if num_workers > 0 else False  # 保持工作进程
        )

    def _compute_sparse_weights(self, coords, max_neighbors=50):
        """计算稀疏空间权重矩阵以减少内存使用"""
        n_samples = len(coords)
        self.logger.info(f"计算稀疏空间权重矩阵，最大邻居数: {max_neighbors}")

        # 使用KD树快速查找最近邻
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)

        # 查找每个点的k个最近邻
        distances, indices = tree.query(coords, k=min(max_neighbors + 1, n_samples))

        # 创建稀疏权重矩阵
        row_indices = []
        col_indices = []
        weight_values = []

        for i in range(n_samples):
            # 排除自身（距离为0）
            mask = distances[i] > 0
            valid_indices = indices[i][mask]
            valid_distances = distances[i][mask]

            # 计算权重（高斯核）
            weights = np.exp(-0.5 * (valid_distances / self.weight_calculator.bandwidth) ** 2)

            # 添加到稀疏矩阵数据
            for j, weight in zip(valid_indices, weights):
                row_indices.append(i)
                col_indices.append(j)
                weight_values.append(weight)

        # 创建稀疏矩阵
        from scipy.sparse import csr_matrix
        sparse_weights = csr_matrix((weight_values, (row_indices, col_indices)),
                                    shape=(n_samples, n_samples))

        self.logger.info(f"稀疏权重矩阵创建完成: {sparse_weights.nnz} 个非零元素")
        return sparse_weights


    def train(self, train_loader, epochs=100, patience=10, validate_spatial=True, pin_memory=True):
        """训练模型 - 内存优化版本"""
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        # 如果使用CUDA，启用cudnn自动优化
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()  # 清空GPU缓存
            initial_memory = torch.cuda.memory_allocated()
            torch.backends.cudnn.benchmark = True

        # # 大样本时禁用空间验证
        # if validate_spatial and self.global_weights is not None:
        #     n_samples = self.global_weights.shape[0] if hasattr(self.global_weights, 'shape') else 0
        #     if n_samples > 5000:
        #         validate_spatial = False
        #         self.logger.info("样本数量较大，禁用空间自相关性验证")

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                if len(batch) == 3:
                    batch_features, batch_targets, batch_coords = batch
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    batch_coords = batch_coords.to(self.device, non_blocking=True)
                else:
                    batch_features, batch_targets = batch
                    batch_features = batch_features.to(self.device, non_blocking=True)
                    batch_targets = batch_targets.to(self.device, non_blocking=True)
                    batch_coords = None


                self.optimizer.zero_grad()

                if self.use_spatial_weights and batch_coords is not None:
                    # GPU上的空间权重计算
                    batch_weights = self._compute_gpu_spatial_weights(batch_coords)
                    outputs = self.model(batch_features, batch_weights, batch_coords)
                    loss = self.spatial_weighted_loss(outputs, batch_targets, batch_weights)
                else:
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)

            # 早停法
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"早停在epoch {epoch}")
                break

            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

            if self.device.type == 'cuda' and epoch % 10 == 0:
                # 监控GPU内存使用
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                cached = torch.cuda.memory_reserved() / 1024 ** 3
                self.logger.info(f"GPU内存 - 已分配: {allocated:.2f}GB, 缓存: {cached:.2f}GB")

    def predict(self, features, coords=None):
        """预测方法"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)

            if self.use_spatial_weights and coords is not None:
                # 如果有坐标数据，计算空间权重
                coords_tensor = torch.FloatTensor(coords).to(self.device)
                # 计算批次内的空间权重
                batch_coords_np = coords_tensor.cpu().numpy()
                batch_weights = self.weight_calculator.calculate_weights(
                    batch_coords_np, batch_coords_np)
                batch_weights = torch.FloatTensor(batch_weights).to(self.device)

                outputs = self.model(features_tensor, batch_weights, coords_tensor)
            else:
                # 没有空间权重
                outputs = self.model(features_tensor)

            return outputs.cpu().numpy().flatten()

    def spatial_weighted_loss(self, outputs, targets, weights):
        """空间加权损失函数"""
        return (weights * (outputs.squeeze() - targets) ** 2).mean()



class EnhancedSpatialDataset(Dataset):
    """增强的空间数据集 - 详细调试版本"""

    def __init__(self, features, targets, coords=None):
        print(f"=== EnhancedSpatialDataset 初始化调试 ===")
        print(f"输入 coords 类型: {type(coords)}")
        print(f"输入 coords is None: {coords is None}")
        print(f"输入 coords id: {id(coords)}")

        if coords is not None:
            print(f"输入 coords 形状: {coords.shape}")
            print(f"输入 coords 数据类型: {coords.dtype}")
        else:
            print("❌ 输入 coords 为 None!")

        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

        print(f"特征转换后 coords 状态: {coords is None}")

        # 直接转换，不添加任何虚拟坐标逻辑
        if coords is not None:
            print("开始坐标转换...")
            self.coords = torch.FloatTensor(coords)
            print("坐标转换成功")
        else:
            # 如果coords为None，直接抛出错误
            raise ValueError("坐标数据不能为None")

        print(f"EnhancedSpatialDataset 初始化完成")
        print(f"最终 coords 形状: {self.coords.shape}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.coords[idx]


# 使用示例
def example_usage():
    # 生成示例数据
    n_samples = 1000
    n_features = 10

    # 生成空间坐标（模拟地理分布）
    coords = np.random.uniform(0, 100, (n_samples, 2))

    # 生成特征和目标（加入空间相关性）
    features = np.random.randn(n_samples, n_features)

    # 创建具有空间相关性的目标变量
    spatial_effect = np.exp(-0.01 * coords[:, 0]) + np.sin(0.1 * coords[:, 1])
    targets = (features[:, 0] + 2 * features[:, 1] + 0.5 * spatial_effect +
               np.random.normal(0, 0.1, n_samples))

    # 创建数据集
    dataset = EnhancedSpatialDataset(features, targets, coords)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 训练模型
    trainer = EnhancedGNNWRTrainer(
        input_dim=n_features,
        coords=coords,
        hidden_dims=[64, 32, 16],
        bandwidth=10.0,  # 带宽参数
        use_spatial_weights=True
    )

    # 训练
    trainer.train(dataloader, epochs=100, patience=15)

    # 预测
    predictions = trainer.predict(features, coords)

    # 计算最终的空间自相关性
    residuals = targets - predictions
    final_moran = trainer.spatial_analyzer.morans_i(residuals, trainer.global_weights)
    print(f"Final Moran's I of residuals: {final_moran:.4f}")

    return trainer, predictions


if __name__ == "__main__":
    trainer, predictions = example_usage()