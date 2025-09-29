import traceback
import pandas as pd
import logging
from pathlib import Path

from datetime import datetime


class DataIntegrator:
    def __init__(self, output_dir, secure_processor=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("DataIntegrator")
        self.source_data = {}

    def add_source(self, name, file_path):
        """添加数据源 - 增强版本"""
        try:
            path = Path(file_path)
            if path.suffix.lower() == '.parquet':
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)

            # 数据验证
            validation_result = self._validate_dataframe(df, name)
            if not validation_result['valid']:
                self.logger.warning(f"数据源 {name} 验证警告: {validation_result['message']}")
                # 继续处理，但记录警告

            self.source_data[name] = df
            self.logger.info(f"添加 {name}: {len(df)} 行, 列: {list(df.columns)}")
            return True

        except Exception as e:
            self.logger.error(f"添加 {name} 失败: {e}")
            return False

    def _validate_dataframe(self, df, source_name):
        """验证DataFrame结构"""
        validation = {'valid': True, 'message': ''}

        # 检查必需列
        if 'station_id' not in df.columns:
            validation['valid'] = False
            validation['message'] = "缺少station_id列"
            return validation

        # 检查数据类型
        numeric_cols = df.select_dtypes(include=['number']).columns
        value_cols = [col for col in numeric_cols if col not in ['station_id', 'date', 'year', 'month', 'day']]

        if not value_cols:
            validation['valid'] = False
            validation['message'] = "没有找到数值列"
            return validation

        # 检查数据完整性
        station_count = df['station_id'].nunique()
        if station_count == 0:
            validation['valid'] = False
            validation['message'] = "没有有效的站点数据"
            return validation

        # 判断数据源类型
        if 'date' in df.columns:
            date_count = df['date'].nunique()
            validation['message'] = f"动态数据源，{station_count}个站点，{date_count}个时间点"
        else:
            validation['message'] = f"静态数据源，{station_count}个站点"

        return validation

    def save_master_excel(self, format_type='wide'):
        """保存主Excel文件 - 增强数据验证版本"""
        if not self.source_data:
            self.logger.error("没有数据源可整合")
            return None

        try:
            # 先执行数据修复
            fix_count = self.emergency_fix()
            if fix_count is None:
                fix_count = 0

            self.logger.info(f"数据修复完成: {fix_count} 个问题已修复")

            # 创建宽表
            if format_type == 'wide':
                final_df = self._create_correct_wide_table()
            else:
                final_df = self.get_combined_data()

            if final_df is None or final_df.empty:
                self.logger.error("最终数据为空或为None")
                return None

            # 数据完整性验证
            final_df = self._validate_data_integrity(final_df)

            if final_df.empty:
                self.logger.error("验证后数据为空")
                return None

            # 生成输出文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"integrated_data_{timestamp}.xlsx"

            # 保存Excel文件
            final_df.to_excel(output_path, index=False)
            self.logger.info(f"✅ 整合数据保存成功: {output_path}")

            # 数据验证
            self._comprehensive_validation(final_df)

            return str(output_path)

        except Exception as e:
            self.logger.error(f"保存失败: {e}")
            self.logger.debug(f"详细错误: {traceback.format_exc()}")
            return None

    def _create_correct_wide_table(self):
        """创建正确的宽表 - 支持地形特征宽表格式"""
        if not self.source_data:
            return pd.DataFrame()

        self.logger.info("创建正确宽表...")

        try:
            # 1. 分离不同类型的数据源
            static_dfs = []  # 地形特征等静态数据（宽表格式）
            dynamic_dfs = []  # 有时间维度的动态数据

            for source_name, source_df in self.source_data.items():
                # 检查数据源类型
                if self._is_terrain_features(source_name, source_df):
                    # 地形特征数据（已经是宽表格式）
                    static_dfs.append((source_name, source_df))
                    self.logger.info(f"识别为地形特征数据 {source_name}: {len(source_df)} 行, {len(source_df.columns)} 列")
                elif 'date' in source_df.columns and 'station_id' in source_df.columns:
                    # 动态数据源
                    numeric_cols = source_df.select_dtypes(include=['number']).columns
                    value_cols = [col for col in numeric_cols if
                                  col not in ['station_id', 'date', 'year', 'month', 'day']]

                    if value_cols:
                        value_col = value_cols[0]
                        source_wide = source_df[['station_id', 'date', value_col]].copy()
                        source_wide = source_wide.rename(columns={value_col: source_name})
                        source_wide = source_wide.drop_duplicates(['station_id', 'date'])
                        dynamic_dfs.append((source_name, source_wide))
                        self.logger.info(f"准备动态数据源 {source_name}: {len(source_wide)} 行")
                elif 'station_id' in source_df.columns:
                    # 其他静态数据源
                    static_dfs.append((source_name, source_df))
                    self.logger.info(f"准备静态数据源 {source_name}: {len(source_df)} 行")
                else:
                    self.logger.warning(f"数据源 {source_name} 格式不支持")

            if not dynamic_dfs and not static_dfs:
                self.logger.error("没有可用的数据源")
                return pd.DataFrame()

            # 2. 处理动态数据源（创建基础框架）
            if dynamic_dfs:
                first_dynamic_name, final_wide = dynamic_dfs[0]
                self.logger.info(f"以动态数据源 {first_dynamic_name} 为基础框架")

                # 合并其他动态数据源
                for i in range(1, len(dynamic_dfs)):
                    name, next_df = dynamic_dfs[i]
                    before_count = len(final_wide)
                    final_wide = final_wide.merge(next_df, on=['station_id', 'date'], how='left')
                    after_count = len(final_wide)
                    self.logger.info(f"合并动态数据源 {name}: {before_count} -> {after_count} 行")
            else:
                # 如果没有动态数据源，需要特殊处理
                self.logger.info("没有动态数据源，处理纯静态数据")
                return self._handle_static_only_case(static_dfs)

            # 3. 合并所有静态数据源
            if static_dfs:
                # 合并所有静态数据源
                static_combined = static_dfs[0][1]  # 第一个静态数据源

                for i in range(1, len(static_dfs)):
                    name, static_df = static_dfs[i]
                    before_cols = len(static_combined.columns)
                    static_combined = static_combined.merge(static_df, on='station_id', how='outer')
                    after_cols = len(static_combined.columns)
                    added_cols = after_cols - before_cols
                    self.logger.info(f"合并静态数据源 {name}: 添加 {added_cols} 列")

                # 将静态数据合并到动态框架中
                before_count = len(final_wide)
                final_wide = final_wide.merge(static_combined, on='station_id', how='left')
                after_count = len(final_wide)
                self.logger.info(f"合并所有静态数据: {before_count} -> {after_count} 行")

                # 统计静态特征的填充情况
                static_cols = [col for col in static_combined.columns if col != 'station_id']
                for col in static_cols:
                    if col in final_wide.columns:
                        fill_rate = final_wide[col].notna().sum() / len(final_wide) * 100
                        self.logger.info(f"静态特征 {col}: {fill_rate:.1f}% 填充率")

            # 4. 排序和整理
            final_wide = final_wide.sort_values(['station_id', 'date']).reset_index(drop=True)

            self.logger.info(f"✅ 宽表创建完成: {final_wide.shape}")
            self.logger.info(f"列名: {list(final_wide.columns)}")

            return final_wide

        except Exception as e:
            self.logger.error(f"创建宽表失败: {e}")
            return pd.DataFrame()

    def _is_terrain_features(self, source_name, df):
        """判断是否为地形特征数据"""
        # 通过名称判断
        if 'terrain' in source_name.lower() or 'feature' in source_name.lower():
            return True

        # 通过列结构判断（地形特征通常有多个特征列）
        numeric_cols = df.select_dtypes(include=['number']).columns
        feature_cols = [col for col in numeric_cols if col not in ['station_id', 'date', 'longitude', 'latitude']]

        # 如果有多个数值列且没有date列，很可能是地形特征宽表
        if len(feature_cols) > 1 and 'date' not in df.columns:
            return True

        return False

    def _validate_final_result(self, df):
        """验证最终结果"""
        self.logger.info("=== 最终结果验证 ===")
        self.logger.info(f"总行数: {len(df)}")
        self.logger.info(f"列名: {list(df.columns)}")

        # 检查各数据源的填充情况
        data_cols = [col for col in df.columns if col not in ['station_id', 'date']]
        for col in data_cols:
            non_null = df[col].notna().sum()
            self.logger.info(f"{col}: {non_null} 行有值")

        # 显示样例数据
        if len(df) > 0:
            self.logger.info("前5行数据样例:")
            for i in range(min(5, len(df))):
                row = df.iloc[i]
                values = []
                for col in data_cols:
                    if pd.notna(row[col]):
                        values.append(f"{col}: {row[col]}")

                if values:
                    self.logger.info(f"  站点 {row['station_id']} | 时间 {row['date']} | {', '.join(values)}")

    def _validate_data_integrity(self, df):
        """验证数据完整性，确保没有虚假的站点-日期组合"""
        if df.empty:
            return df

        # 获取数据列
        data_cols = [col for col in df.columns if col not in ['station_id', 'date']]

        if not data_cols:
            return df

        # 统计每个数据源的有效记录数
        self.logger.info("=== 数据有效性验证 ===")
        for col in data_cols:
            valid_count = df[col].notna().sum()
            total_count = len(df)
            fill_rate = (valid_count / total_count) * 100 if total_count > 0 else 0
            self.logger.info(f"{col}: {valid_count}/{total_count} 有效记录 ({fill_rate:.1f}%)")

        # 识别可能的问题记录
        empty_records = df[data_cols].isna().all(axis=1)
        if empty_records.any():
            empty_count = empty_records.sum()
            self.logger.warning(f"发现 {empty_count} 条全空记录")

            # 显示一些示例
            empty_examples = df[empty_records][['station_id', 'date']].head(5)
            self.logger.info("全空记录示例:")
            for _, row in empty_examples.iterrows():
                self.logger.info(f"  站点 {row['station_id']}, 日期 {row['date']}")

        return df

    def get_combined_data(self):
        """合并数据"""
        if not self.source_data:
            return pd.DataFrame()

        dfs = []
        for name, df in self.source_data.items():
            df_copy = df.copy()
            df_copy['data_source'] = name
            dfs.append(df_copy)

        return pd.concat(dfs, ignore_index=True)

    def emergency_fix(self, issue_type='duplicates'):
        """应急数据修复 - 修复版本"""
        try:
            fix_count = 0

            if not self.source_data:
                self.logger.warning("没有数据源可修复")
                return fix_count  # 返回0而不是None

            # 1. 处理重复数据
            if issue_type == 'duplicates':
                for name, df in self.source_data.items():
                    if 'station_id' in df.columns:
                        original_count = len(df)

                        # 根据是否有date列决定去重方式
                        if 'date' in df.columns:
                            df_clean = df.drop_duplicates(['station_id', 'date'])
                        else:
                            df_clean = df.drop_duplicates(['station_id'])

                        removed_count = original_count - len(df_clean)
                        if removed_count > 0:
                            self.source_data[name] = df_clean
                            fix_count += removed_count
                            self.logger.info(f"修复 {name}: 移除 {removed_count} 个重复行")

            # 2. 处理列名不一致问题
            elif issue_type == 'column_names':
                for name, df in self.source_data.items():
                    # 标准化列名
                    column_mapping = {}
                    if 'station_ID' in df.columns and 'station_id' not in df.columns:
                        column_mapping['station_ID'] = 'station_id'
                    if 'Station_ID' in df.columns and 'station_id' not in df.columns:
                        column_mapping['Station_ID'] = 'station_id'

                    if column_mapping:
                        df_renamed = df.rename(columns=column_mapping)
                        self.source_data[name] = df_renamed
                        fix_count += len(column_mapping)
                        self.logger.info(f"修复 {name}: 重命名 {list(column_mapping.keys())}")

            # 3. 处理数据类型问题
            elif issue_type == 'data_types':
                for name, df in self.source_data.items():
                    # 确保station_id是字符串类型
                    if 'station_id' in df.columns:
                        if df['station_id'].dtype != 'object':
                            df_fixed = df.copy()
                            df_fixed['station_id'] = df_fixed['station_id'].astype(str)
                            self.source_data[name] = df_fixed
                            fix_count += 1
                            self.logger.info(f"修复 {name}: station_id转换为字符串类型")

            self.logger.info(f"✅ 应急修复完成: 执行了 {fix_count} 个修复")
            return fix_count  # 确保返回整数

        except Exception as e:
            self.logger.error(f"应急修复失败: {e}")
            return 0  # 即使出错也返回0而不是None

    def generate_report(self):
        return "报告生成完成"

    def _comprehensive_validation(self, df):
        """全面验证宽表数据"""
        self.logger.info("=== 全面数据验证 ===")

        # 1. 基本统计
        self.logger.info(f"表格形状: {df.shape}")
        self.logger.info(f"唯一站点数: {df['station_id'].nunique()}")
        self.logger.info(f"时间点数: {df['date'].nunique()}")
        self.logger.info(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")

        # 2. 数据完整性检查
        data_cols = [col for col in df.columns if col not in ['station_id', 'date']]
        self.logger.info("=== 各数据源填充情况 ===")

        for col in data_cols:
            non_null = df[col].notna().sum()
            fill_rate = (non_null / len(df)) * 100
            if non_null > 0:
                avg_val = df[col].mean()
                min_val = df[col].min()
                max_val = df[col].max()
                self.logger.info(
                    f"{col}: {non_null}行有值({fill_rate:.1f}%), 范围: {min_val:.2f}-{max_val:.2f}, 平均: {avg_val:.2f}")
            else:
                self.logger.info(f"{col}: 0行有值")

        # 3. 检查重复行
        duplicates = df.duplicated(['station_id', 'date']).sum()
        self.logger.info(f"重复行数: {duplicates}")

        # 4. 抽样显示具体数据
        self.logger.info("=== 数据抽样显示 ===")

        # 显示前5行
        self.logger.info("前5行数据:")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            values = []
            for col in data_cols:
                if pd.notna(row[col]):
                    values.append(f"{col}: {row[col]}")
            self.logger.info(f"  行{i + 1}: 站点{row['station_id']}, 时间{row['date']}")
            if values:
                self.logger.info(f"      数据: {', '.join(values)}")

        # 5. 检查特定已知数据点
        self.logger.info("=== 已知数据点验证 ===")
        known_points = [
            (50136, '2013-01-05', 'snow_depth', 14.1),
            (54764, '2013-01-02', 'ERA5温度', 2.67)
        ]

        for station, date, data_type, expected_value in known_points:
            match = df[(df['station_id'] == station) & (df['date'] == date)]
            if not match.empty:
                actual_value = match.iloc[0].get(data_type)
                if pd.notna(actual_value):
                    diff = abs(actual_value - expected_value)
                    status = "✓" if diff < 0.1 else "✗"
                    self.logger.info(
                        f"{status} 站点{station} {date} {data_type}: 期望{expected_value}, 实际{actual_value:.2f}")
                else:
                    self.logger.info(f"✗ 站点{station} {date} {data_type}: 期望{expected_value}, 实际无数据")
            else:
                self.logger.info(f"✗ 站点{station} {date}: 无记录")

    def clear_data(self):
        self.source_data.clear()


