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
        features = self.feature_network(x)

        # 空间注意力加权
        if self.use_attention and spatial_weights is not None and coords is not None:
            # 应用空间权重
            weighted_features = features * spatial_weights.unsqueeze(1)
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
    """增强的GNNWR训练器"""

    def __init__(self, input_dim, coords, hidden_dims=[64, 32, 16],
                 learning_rate=0.001, bandwidth=None, use_spatial_weights=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.coords = coords
        self.use_spatial_weights = use_spatial_weights

        # 初始化空间权重计算器
        self.weight_calculator = SpatialWeightCalculator(bandwidth=bandwidth)
        if bandwidth is None:
            # 自动计算自适应带宽
            bandwidth = self.weight_calculator.adaptive_bandwidth(coords)
            self.weight_calculator.bandwidth = bandwidth

        # 计算全局空间权重矩阵
        self.global_weights = self.weight_calculator.calculate_weights(coords, coords)

        # 初始化模型
        self.model = EnhancedGNNWRModel(input_dim, hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        # 空间自相关性分析
        self.spatial_analyzer = SpatialAutocorrelation()

    def spatial_weighted_loss(self, outputs, targets, weights):
        """空间加权损失函数"""
        return (weights * (outputs - targets) ** 2).mean()

    def train(self, train_loader, epochs=100, patience=10, validate_spatial=True):
        """训练模型"""
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        spatial_stats = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            all_predictions = []
            all_targets = []

            for batch in train_loader:
                if len(batch) == 4:  # 有空间权重
                    batch_features, batch_targets, batch_coords, batch_weights = batch
                    batch_weights = batch_weights.to(self.device)
                else:  # 没有空间权重
                    batch_features, batch_targets, batch_coords = batch
                    batch_weights = None

                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_coords = batch_coords.to(self.device)

                self.optimizer.zero_grad()

                if self.use_spatial_weights and batch_weights is not None:
                    outputs = self.model(batch_features, batch_weights, batch_coords)
                    loss = self.spatial_weighted_loss(outputs, batch_targets, batch_weights)
                else:
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_targets)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                all_predictions.extend(outputs.detach().cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())

            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)

            # 空间自相关性分析
            if validate_spatial and epoch % 10 == 0:
                residuals = np.array(all_targets) - np.array(all_predictions)
                morans_i = self.spatial_analyzer.morans_i(residuals, self.global_weights)
                spatial_stats.append(morans_i)
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}, Moran's I: {morans_i:.4f}")
            else:
                print(f"Epoch {epoch}, Loss: {epoch_loss:.6f}")

            # 早停法
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_gnnwr_model.pth')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 绘制训练曲线和空间统计
        self._plot_training_stats(train_losses, spatial_stats)

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_gnnwr_model.pth'))

    def predict(self, features, coords=None):
        """预测 - 支持不同位置预测"""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)

            if coords is not None and self.use_spatial_weights:
                # 计算预测位置的空间权重
                predict_weights = self.weight_calculator.calculate_weights(
                    self.coords, coords)
                weights_tensor = torch.FloatTensor(predict_weights).to(self.device)
                coords_tensor = torch.FloatTensor(coords).to(self.device)

                predictions = []
                for i in range(len(features)):
                    pred = self.model(features_tensor[i:i + 1],
                                      weights_tensor[:, i:i + 1],
                                      coords_tensor[i:i + 1])
                    predictions.append(pred.item())
                return np.array(predictions)
            else:
                predictions = self.model(features_tensor)
                return predictions.cpu().numpy()

    def _plot_training_stats(self, losses, spatial_stats):
        """绘制训练统计图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # 损失曲线
        ax1.plot(losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.grid(True)

        # 空间自相关性
        if spatial_stats:
            ax2.plot(range(0, len(losses), 10)[:len(spatial_stats)], spatial_stats)
            ax2.set_title("Moran's I of Residuals")
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel("Moran's I")
            ax2.grid(True)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('training_stats.png', dpi=300, bbox_inches='tight')
        plt.close()


class SpatialDataset(Dataset):
    """空间数据集"""

    def __init__(self, features, targets, spatial_weights=None):
        """
        Args:
            features (np.array): 特征数据
            targets (np.array): 目标变量
            spatial_weights (np.array, optional): 空间权重
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.spatial_weights = torch.FloatTensor(spatial_weights) if spatial_weights is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.spatial_weights is not None:
            return self.features[idx], self.targets[idx], self.spatial_weights[idx]
        else:
            return self.features[idx], self.targets[idx]


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