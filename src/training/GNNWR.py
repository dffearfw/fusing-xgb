import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
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
    """优化的GNNWR模型 - 精度优先版本"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], output_dim=1,
                 dropout_rate=0.3, use_attention=True):
        super(EnhancedGNNWRModel, self).__init__()
        self.use_attention = use_attention

        # 增强的特征处理网络
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.feature_network = nn.Sequential(*layers)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate // 2),
            nn.Linear(prev_dim // 2, output_dim)
        )

        # 增强的空间注意力
        if use_attention:
            self.spatial_attention = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, x, spatial_weights=None, coords=None):
        # 特征提取
        features = self.feature_network(x)

        # 空间注意力加权
        if self.use_attention and spatial_weights is not None and coords is not None:
            # 空间平滑
            row_sums = torch.sum(spatial_weights, dim=1, keepdim=True)
            normalized_weights = spatial_weights / torch.where(row_sums > 0, row_sums, torch.tensor(1.0))
            weighted_features = torch.matmul(normalized_weights, features)
            output = self.output_layer(weighted_features)
        else:
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

    def __init__(self, features, targets, coords=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)

        if coords is not None:
            self.coords = torch.FloatTensor(coords)
        else:
            self.coords = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.coords is not None:
            return self.features[idx], self.targets[idx], self.coords[idx]
        else:
            return self.features[idx], self.targets[idx]


class EnhancedGNNWRTrainer:
    """增强的GNNWR训练器 - GPU混合精度优化版本"""

    def __init__(self, input_dim, coords, hidden_dims=[256, 128, 64, 32],
                 learning_rate=0.001, bandwidth=None, use_spatial_weights=True,
                 max_samples_for_spatial=20000, device='auto',
                 mixed_precision=True, cpu_workers=24):

        # 添加 logger 初始化
        self.logger = logging.getLogger("EnhancedGNNWRTrainer")

        # 混合精度设置
        self.mixed_precision = mixed_precision and torch.cuda.is_available()
        if self.mixed_precision:
            self.scaler = GradScaler()
            self.logger.info("启用混合精度训练")

        # 自动检测或指定设备
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # GPU优化设置
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                self.logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = torch.device('cpu')
                torch.set_num_threads(cpu_workers)
                self.logger.info(f"使用CPU: {cpu_workers}线程")
        else:
            self.device = torch.device(device)
            if device == 'cpu':
                torch.set_num_threads(cpu_workers)

        self.logger.info(f"使用设备: {self.device}")

        # 坐标处理
        if coords is not None:
            self.coords = coords.copy()  # 创建副本，避免修改原数据
        else:
            self.coords = None

        self.use_spatial_weights = use_spatial_weights and (self.coords is not None)

        # 检查样本数量，如果太多则禁用空间权重
        if self.coords is not None and len(self.coords) > max_samples_for_spatial:
            self.logger.warning(f"样本数量 {len(self.coords)} 超过限制 {max_samples_for_spatial}，禁用空间权重")
            self.use_spatial_weights = False

        # 初始化空间权重计算器（仅在需要时）
        if self.use_spatial_weights:
            self.weight_calculator = SpatialWeightCalculator(bandwidth=bandwidth)
            if bandwidth is None:
                # 使用较小的k值以减少计算量
                bandwidth = self.weight_calculator.adaptive_bandwidth(self.coords, k=5)
                self.weight_calculator.bandwidth = bandwidth

            # 计算全局空间权重矩阵（分批计算以避免内存问题）
            self.global_weights = self._compute_sparse_weights(self.coords)
        else:
            self.weight_calculator = None
            self.global_weights = None

        # 初始化模型 - 只初始化一次！
        self.model = EnhancedGNNWRModel(input_dim, hidden_dims).to(self.device)

        # 优化器 - 针对混合精度优化
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=200,
            steps_per_epoch=1000,
            pct_start=0.1
        )

        self.criterion = nn.HuberLoss()  # 更稳定的损失函数

        self.logger.info(f"EnhancedGNNWR训练器初始化完成，使用空间权重: {self.use_spatial_weights}")

    def create_optimized_dataloader(self, dataset, batch_size=256, pin_memory=True):
        """创建优化的数据加载器，充分利用14900KF"""
        num_workers = min(16, os.cpu_count() - 4)  # 为系统保留4个核心

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and self.device.type == 'cuda',
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory_device=str(self.device) if pin_memory else None
        )

    def _compute_gpu_spatial_weights(self, batch_coords):
        """在GPU上计算空间权重 - 混合精度优化"""
        n_batch = batch_coords.shape[0]

        if n_batch <= 1:
            return torch.ones((n_batch, n_batch), device=self.device,
                              dtype=torch.float16 if self.mixed_precision else torch.float32)

        # 使用混合精度计算
        with autocast(device_type='cuda',enabled=self.mixed_precision):
            # 使用PyTorch向量化计算（GPU加速）
            diff = batch_coords.unsqueeze(1) - batch_coords.unsqueeze(0)
            distances = torch.sqrt(torch.sum(diff ** 2, dim=2) + 1e-8)

            # 自适应带宽
            sorted_distances = torch.sort(distances, dim=1).values
            bandwidth = torch.mean(sorted_distances[:, 1:6])  # 前5个邻居

            # 计算权重
            weights = torch.exp(-0.5 * (distances / bandwidth) ** 2)

            # 归一化
            row_sums = torch.sum(weights, dim=1, keepdim=True)
            weights = weights / torch.where(row_sums > 0, row_sums, torch.tensor(1.0))

        return weights

    def _compute_cpu_spatial_weights(self, batch_coords):
        """CPU版本的空间权重计算"""
        n_batch = batch_coords.shape[0]
        if n_batch <= 1:
            return torch.ones((n_batch, n_batch), device=self.device)

        # 使用numpy计算（CPU上更快）
        batch_coords_np = batch_coords.cpu().numpy()
        distances = cdist(batch_coords_np, batch_coords_np, metric='euclidean')
        weights = np.exp(-0.5 * (distances / self.weight_calculator.bandwidth) ** 2)

        # 转换为tensor
        weights_tensor = torch.FloatTensor(weights).to(self.device)

        # 归一化
        row_sums = torch.sum(weights_tensor, dim=1, keepdim=True)
        normalized_weights = weights_tensor / torch.where(row_sums > 0, row_sums, torch.tensor(1.0))

        return normalized_weights

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

    def train_epoch_mixed_precision(self, train_loader):
        """混合精度训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # 数据移动到设备（异步传输）
            if len(batch) == 3:
                features, targets, batch_coords = batch
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_coords = batch_coords.to(self.device, non_blocking=True) if batch_coords is not None else None
            else:
                features, targets = batch
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                batch_coords = None

            self.optimizer.zero_grad(set_to_none=True)  # 更快的梯度清零

            # 混合精度前向传播
            with autocast(device_type='cuda',enabled=self.mixed_precision):
                if self.use_spatial_weights and batch_coords is not None:
                    if self.device.type == 'cuda':
                        batch_weights = self._compute_gpu_spatial_weights(batch_coords)
                    else:
                        batch_weights = self._compute_cpu_spatial_weights(batch_coords)
                    outputs = self.model(features, batch_weights, batch_coords)
                else:
                    outputs = self.model(features)

                loss = self.criterion(outputs, targets)

            # 混合精度反向传播
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # 学习率调度
            self.scheduler.step()

            total_loss += loss.item()

            # 每100个batch输出一次进度
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(f'Batch {batch_idx}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}')

        return total_loss / len(train_loader)

    def train(self, train_loader, epochs=200, patience=20, val_loader=None):
        """训练模型 - 混合精度优化版本"""
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        # GPU预热
        if self.device.type == 'cuda':
            self._warmup_gpu()

        for epoch in range(epochs):
            # 训练阶段
            epoch_loss = self.train_epoch_mixed_precision(train_loader)
            train_losses.append(epoch_loss)

            # 验证阶段
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)

                self.logger.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

                # 早停检查
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_gnnwr_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(f'早停在epoch {epoch + 1}')
                    break
            else:
                # 没有验证集的情况
                self.logger.info(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.6f}')

                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_gnnwr_model.pth')
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    self.logger.info(f'早停在epoch {epoch + 1}')
                    break

        # 加载最佳模型
        if os.path.exists('best_gnnwr_model.pth'):
            self.model.load_state_dict(torch.load('best_gnnwr_model.pth'))
            self.logger.info("加载最佳模型完成")

        return train_losses, val_losses if val_loader is not None else train_losses

    def validate(self, val_loader):
        """验证阶段 - 混合精度优化"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                # 数据移动到设备
                if len(batch) == 3:
                    features, targets, batch_coords = batch
                    features = features.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    batch_coords = batch_coords.to(self.device, non_blocking=True) if batch_coords is not None else None
                else:
                    features, targets = batch
                    features = features.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    batch_coords = None

                # 混合精度前向传播
                with autocast(enabled=self.mixed_precision):
                    if self.use_spatial_weights and batch_coords is not None:
                        if self.device.type == 'cuda':
                            batch_weights = self._compute_gpu_spatial_weights(batch_coords)
                        else:
                            batch_weights = self._compute_cpu_spatial_weights(batch_coords)
                        outputs = self.model(features, batch_weights, batch_coords)
                    else:
                        outputs = self.model(features)

                    loss = self.criterion(outputs, targets)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _warmup_gpu(self):
        """GPU预热 - 针对5080优化"""
        if self.device.type == 'cuda':
            self.logger.info("进行GPU预热...")
            # 运行一个小的虚拟计算来预热GPU
            dummy_input = torch.randn(32, self.model.feature_network[0].in_features,
                                      device=self.device,
                                      dtype=torch.float16 if self.mixed_precision else torch.float32)
            dummy_coords = torch.randn(32, 2, device=self.device,
                                       dtype=torch.float16 if self.mixed_precision else torch.float32)

            with autocast(device_type='cuda',enabled=self.mixed_precision):
                for _ in range(10):
                    if self.use_spatial_weights:
                        dummy_weights = self._compute_gpu_spatial_weights(dummy_coords)
                        _ = self.model(dummy_input, dummy_weights, dummy_coords)
                    else:
                        _ = self.model(dummy_input)

            torch.cuda.synchronize()
            self.logger.info("GPU预热完成")

    def predict(self, features, coords=None, batch_size=1024):
        """批量预测 - 内存和性能优化"""
        self.model.eval()
        predictions = []

        # 确定合适的数据类型
        dtype = torch.float16 if self.mixed_precision else torch.float32

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                # 分批处理避免内存溢出
                end_idx = min(i + batch_size, len(features))
                batch_features = torch.tensor(features[i:end_idx], dtype=dtype, device=self.device)

                batch_coords = None
                if coords is not None:
                    batch_coords = torch.tensor(coords[i:end_idx], dtype=dtype, device=self.device)

                # 混合精度预测
                with autocast(device_type='cuda',enabled=self.mixed_precision):
                    if self.use_spatial_weights and batch_coords is not None:
                        if self.device.type == 'cuda':
                            batch_weights = self._compute_gpu_spatial_weights(batch_coords)
                        else:
                            batch_weights = self._compute_cpu_spatial_weights(batch_coords)
                        batch_pred = self.model(batch_features, batch_weights, batch_coords)
                    else:
                        batch_pred = self.model(batch_features)

                predictions.append(batch_pred.cpu().numpy())

        return np.concatenate(predictions)

    def spatial_weighted_loss(self, outputs, targets, weights):
        """空间加权损失函数"""
        return (weights * (outputs.squeeze() - targets) ** 2).mean()

    def get_training_info(self):
        """获取训练信息"""
        info = {
            'device': str(self.device),
            'mixed_precision': self.mixed_precision,
            'use_spatial_weights': self.use_spatial_weights,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_layers': len(list(self.model.children()))
        }

        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB",
                'cudnn_enabled': torch.backends.cudnn.enabled
            })

        return info


class DistributedGNNWRTrainer:
    """分布式训练器 - 多GPU支持"""

    def __init__(self, input_dim, coords, hidden_dims=[256, 128, 64, 32],
                 learning_rate=0.001, bandwidth=None, use_spatial_weights=True,
                 mixed_precision=True):
        self.logger = logging.getLogger("DistributedGNNWRTrainer")

        # 分布式设置
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f'cuda:{self.local_rank}')

        # 混合精度
        self.mixed_precision = mixed_precision
        if self.mixed_precision:
            self.scaler = GradScaler()

        # 模型初始化
        self.model = EnhancedGNNWRModel(input_dim, hidden_dims).to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.HuberLoss()

        self.logger.info(f"分布式训练器初始化完成 - Rank: {self.rank}, GPU: {self.local_rank}")


# 使用示例和测试函数
def optimized_example_usage():
    """优化后的使用示例"""
    # 生成示例数据
    n_samples = 10000
    n_features = 20

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

    # 创建优化训练器 - 针对5080 + 14900KF优化
    trainer = EnhancedGNNWRTrainer(
        input_dim=n_features,
        coords=coords,
        hidden_dims=[512, 256, 128, 64],  # 更大的模型充分利用5080
        learning_rate=0.001,
        bandwidth=10.0,
        use_spatial_weights=True,
        device='cuda',  # 使用GPU
        mixed_precision=True,  # 启用混合精度
        cpu_workers=24,  # 充分利用14900KF
        max_samples_for_spatial=50000  # 提高样本限制
    )

    # 显示训练信息
    if hasattr(trainer, 'get_training_info'):
        training_info = trainer.get_training_info()
    else:
        # 提供默认值或使用其他方法获取信息
        training_info = {
            'status': 'training_in_progress',
            'message': 'Training info not available'
        }

    print("=== 训练配置信息 ===")
    for key, value in training_info.items():
        print(f"{key}: {value}")

    # 创建优化的数据加载器
    train_loader = trainer.create_optimized_dataloader(
        dataset,
        batch_size=512,  # 5080可以处理更大的批次
        pin_memory=True
    )

    # 开始训练
    train_losses, val_losses = trainer.train(
        train_loader,
        epochs=200,
        patience=20
    )

    # 预测
    predictions = trainer.predict(features, coords, batch_size=2048)  # 更大的批次预测

    # 计算性能指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr

    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r_value, _ = pearsonr(targets, predictions)

    print(f"\n=== 模型性能 ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R: {r_value:.4f}")

    return trainer, predictions


def benchmark_training():
    """训练性能基准测试"""
    import time
    from tqdm import tqdm

    # 测试数据
    n_samples = 5000
    n_features = 50
    coords = np.random.uniform(0, 100, (n_samples, 2))
    features = np.random.randn(n_samples, n_features)
    targets = np.random.randn(n_samples)

    dataset = EnhancedSpatialDataset(features, targets, coords)

    # 测试不同配置
    configs = [
        {'device': 'cpu', 'mixed_precision': False, 'batch_size': 128},
        {'device': 'cuda', 'mixed_precision': False, 'batch_size': 512},
        {'device': 'cuda', 'mixed_precision': True, 'batch_size': 512}
    ]

    results = {}

    for config in configs:
        print(f"\n测试配置: {config}")

        trainer = EnhancedGNNWRTrainer(
            input_dim=n_features,
            coords=coords,
            device=config['device'],
            mixed_precision=config['mixed_precision']
        )

        train_loader = trainer.create_optimized_dataloader(
            dataset,
            batch_size=config['batch_size']
        )

        # 预热
        if config['device'] == 'cuda':
            trainer._warmup_gpu()

        # 基准测试
        start_time = time.time()

        trainer.model.train()
        for batch in tqdm(train_loader, desc=f"Training {config['device']}"):
            if len(batch) == 3:
                features, targets, batch_coords = batch
                features = features.to(trainer.device)
                targets = targets.to(trainer.device)
                batch_coords = batch_coords.to(trainer.device)
            else:
                features, targets = batch
                features = features.to(trainer.device)
                targets = targets.to(trainer.device)
                batch_coords = None

            trainer.optimizer.zero_grad()

            with autocast(enabled=config['mixed_precision']):
                if trainer.use_spatial_weights and batch_coords is not None:
                    batch_weights = trainer._compute_gpu_spatial_weights(batch_coords)
                    outputs = trainer.model(features, batch_weights, batch_coords)
                else:
                    outputs = trainer.model(features)

                loss = trainer.criterion(outputs, targets)

            if config['mixed_precision']:
                trainer.scaler.scale(loss).backward()
                trainer.scaler.step(trainer.optimizer)
                trainer.scaler.update()
            else:
                loss.backward()
                trainer.optimizer.step()

        end_time = time.time()
        results[str(config)] = end_time - start_time
        print(f"训练时间: {end_time - start_time:.2f}秒")

    print("\n=== 性能对比 ===")
    for config, time_taken in results.items():
        print(f"{config}: {time_taken:.2f}秒")

    return results


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== GNNWR GPU混合精度训练测试 ===")

    # 运行优化示例
    trainer, predictions = optimized_example_usage()

    # 运行性能基准测试
    benchmark_results = benchmark_training()

    print("测试完成！")