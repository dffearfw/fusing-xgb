import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, file_path):
        """
        初始化数据预处理器
        """
        self.file_path = file_path
        self.data = None
        self.numeric_columns = []
        self.categorical_columns = []

    def load_data(self):
        """加载Excel数据"""
        try:
            self.data = pd.read_excel(self.file_path)
            print(f"数据加载成功，形状: {self.data.shape}")
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False

    def analyze_data(self):
        """分析数据特征"""
        print("\n=== 数据基本信息 ===")
        print(f"数据形状: {self.data.shape}")
        print(f"列名: {list(self.data.columns)}")

        print("\n=== 数据类型分布 ===")
        print(self.data.dtypes.value_counts())

        print("\n=== 缺失值统计 ===")
        missing_stats = self.data.isnull().sum()
        missing_percent = (missing_stats / len(self.data)) * 100
        missing_info = pd.DataFrame({
            '缺失数量': missing_stats,
            '缺失比例%': missing_percent
        })
        print(missing_info[missing_info['缺失数量'] > 0])

        # 分类数值列
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()

        print(f"\n数值列: {self.numeric_columns}")
        print(f"分类列: {self.categorical_columns}")

    def handle_missing_values(self, strategy='auto'):
        """
        处理缺失值
        strategy: 'auto', 'mean', 'median', 'mode', 'knn', 'iterative'
        """
        print("\n=== 开始处理缺失值 ===")

        data_clean = self.data.copy()

        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                missing_count = self.data[column].isnull().sum()
                missing_percent = (missing_count / len(self.data)) * 100
                print(f"处理列 '{column}': 缺失 {missing_count} 个值 ({missing_percent:.2f}%)")

                # 数值型数据缺失值处理
                if column in self.numeric_columns:
                    data_clean = self._handle_numeric_missing(data_clean, column, strategy)

                # 分类型数据缺失值处理
                elif column in self.categorical_columns:
                    data_clean = self._handle_categorical_missing(data_clean, column)

        self.data = data_clean
        print("缺失值处理完成")

    def _handle_numeric_missing(self, data, column, strategy):
        """处理数值型数据缺失值"""
        if strategy == 'auto':
            # 根据缺失比例选择策略
            missing_ratio = data[column].isnull().sum() / len(data)

            if missing_ratio < 0.05:
                # 缺失较少，使用中位数
                imputer = SimpleImputer(strategy='median')
            elif missing_ratio < 0.3:
                # 中等缺失，使用KNN
                imputer = KNNImputer(n_neighbors=5)
            else:
                # 大量缺失，使用多重插补
                imputer = IterativeImputer(max_iter=10, random_state=42)
        else:
            strategy_map = {
                'mean': SimpleImputer(strategy='mean'),
                'median': SimpleImputer(strategy='median'),
                'knn': KNNImputer(n_neighbors=5),
                'iterative': IterativeImputer(max_iter=10, random_state=42)
            }
            imputer = strategy_map.get(strategy, SimpleImputer(strategy='median'))

        # 执行插补
        data[column] = imputer.fit_transform(data[[column]]).ravel()
        return data

    def _handle_categorical_missing(self, data, column):
        """处理分类型数据缺失值"""
        # 使用众数填充
        mode_value = data[column].mode()
        if len(mode_value) > 0:
            data[column].fillna(mode_value[0], inplace=True)
        else:
            # 如果没有众数，使用"Unknown"
            data[column].fillna('Unknown', inplace=True)
        return data

    def validate_data(self):
        """验证数据完整性"""
        print("\n=== 数据验证 ===")
        remaining_missing = self.data.isnull().sum().sum()
        if remaining_missing == 0:
            print("✓ 所有缺失值已成功处理")
        else:
            print(f"⚠ 仍有 {remaining_missing} 个缺失值")

        print(f"最终数据形状: {self.data.shape}")

    def save_data(self, output_path):
        """保存处理后的数据"""
        try:
            self.data.to_excel(output_path, index=False)
            print(f"处理后的数据已保存至: {output_path}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False


# 使用示例
def main():
    # 初始化预处理器
    preprocessor = DataPreprocessor('../training/lu_onehot.xlsx')

    # 执行预处理流程
    if preprocessor.load_data():
        preprocessor.analyze_data()
        preprocessor.handle_missing_values(strategy='auto')
        preprocessor.validate_data()
        preprocessor.save_data('processed_data.xlsx')


if __name__ == "__main__":
    main()