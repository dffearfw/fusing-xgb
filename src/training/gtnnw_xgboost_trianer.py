class GTNNW_XGBoostTrainer:
    """GTNNW-XGBoost训练器 - 集成GTNNWR权重矩阵与XGBoost"""

    # 默认XGBoost参数
    DEFAULT_PARAMS = {
        'n_estimators': 60,
        'learning_rate': 0.17,
        'max_depth': 5,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.05,
        'random_state': 42,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # GTNNWR参数
    DEFAULT_GTNNWR_PARAMS = {
        'dense_layers': [[3], [512, 256, 64]],
        'drop_out': 0.4,
        'optimizer': "Adadelta",
        'optimizer_params': {
            "scheduler": "MultiStepLR",
            "scheduler_milestones": [1000, 2000, 3000, 4000],
            "scheduler_gamma": 0.8,
        },
        'max_epoch': 3000,
        'early_stop': 1000,
        'print_frequency': 100
    }

    def __init__(self, params=None, gtnnwr_params=None, use_gtnnwr=True,
                 nan_strategy='median', nan_fill_value=0.0,
                 # 新增参数
                 use_feature_mahalanobis=False,
                 feature_columns_for_distance=None):
        """初始化训练器

        Args:
            use_feature_mahalanobis: 是否使用特征马氏距离
            feature_columns_for_distance: 用于马氏距离计算的特征列
        """
        self.logger = logger
        self.model = None
        self.feature_columns = None
        self.target_column = 'swe'
        self.use_gtnnwr = use_gtnnwr
        self.nan_strategy = nan_strategy
        self.nan_fill_value = nan_fill_value

        # 新增：特征马氏距离相关参数
        self.use_feature_mahalanobis = use_feature_mahalanobis
        self.feature_columns_for_distance = feature_columns_for_distance

        # 存储填充值用于后续预测
        self.nan_fill_values = {}
        self.nan_fill_stats = {}

        # 定义GTNNWR特征列
        self.gtnnwr_x_columns = ['aspect', 'slope', 'eastness', 'tpi', 'curvature1', 'curvature2', 'elevation',
                                 'std_slope', 'std_eastness', 'std_tpi', 'std_curvature1', 'std_curvature2', 'std_high',
                                 'std_aspect', 'glsnow', 'cswe', 'snow_depth_snow_depth', 'ERA5温度_ERA5温度',
                                 'era5_swe', 'doy',
                                 'gldas', 'year', 'month', 'scp_start', 'scp_end', 'd1', 'd2', 'X', 'Y', 'Z', 'da',
                                 'db', 'dc',
                                 'dd']

        # GTNNWR需要空间列和时间列
        self.gtnnwr_spatial_columns = ['X', 'Y']
        self.gtnnwr_temp_columns = ['year', 'month', 'doy']
        self.gtnnwr_id_column = 'id'
        self.gtnnwr_y_column = ['swe']

        # 更新参数
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.gtnnwr_params = self.DEFAULT_GTNNWR_PARAMS.copy()
        if gtnnwr_params:
            self.gtnnwr_params.update(gtnnwr_params)

        self.logger.info(f"初始化GTNNW-XGBoost训练器")
        self.logger.info(f"XGBoost参数: {self.params}")
        self.logger.info(f"使用GTNNWR权重增强: {self.use_gtnnwr}")
        self.logger.info(f"使用特征马氏距离: {self.use_feature_mahalanobis}")
        self.logger.info(f"NaN处理策略: {self.nan_strategy}")

    def _train_gtnnwr_for_fold(self, train_data, val_data):
        """为单个折叠训练GTNNWR模型并提取权重"""
        self.logger.debug("为当前折叠训练GTNNWR模型...")

        print("\n" + "=" * 80)
        print("🧠 GTNNWR模型训练 (当前折叠)")
        print("=" * 80)

        try:
            # 确保所有需要的列都存在
            print("🔍 检查数据完整性...")
            required_columns = (self.gtnnwr_x_columns + self.gtnnwr_spatial_columns +
                                self.gtnnwr_temp_columns + [self.gtnnwr_id_column] + self.gtnnwr_y_column)

            # 检查数据量是否足够
            if len(train_data) < 10 or len(val_data) < 1:
                print(f"⚠️  数据量不足: 训练集{len(train_data)}样本, 验证集{len(val_data)}样本")
                print("⚠️  跳过GTNNWR训练，返回None权重")
                return None, None

            for col in required_columns:
                if col not in train_data.columns:
                    if col == 'id':
                        train_data[col] = np.arange(len(train_data))
                    else:
                        train_data[col] = 0.0
                    print(f"  ⚠️  训练数据缺失列 '{col}'，已填充")
                if col not in val_data.columns:
                    if col == 'id':
                        val_data[col] = np.arange(len(val_data))
                    else:
                        val_data[col] = 0.0
                    print(f"  ⚠️  验证数据缺失列 '{col}'，已填充")

            # 检查数据形状
            print(f"📊 数据形状:")
            print(f"  训练数据: {train_data.shape}")
            print(f"  验证数据: {val_data.shape}")

            # 检查NaN值
            train_nan = train_data[self.gtnnwr_x_columns].isna().sum().sum()
            val_nan = val_data[self.gtnnwr_x_columns].isna().sum().sum()
            if train_nan > 0 or val_nan > 0:
                print(f"  ⚠️  警告: 训练数据有{train_nan}个NaN，验证数据有{val_nan}个NaN")
                # 使用中位数填充
                for col in self.gtnnwr_x_columns:
                    if col in train_data.columns:
                        median_val = train_data[col].median()
                        train_data[col] = train_data[col].fillna(median_val)
                        val_data[col] = val_data[col].fillna(median_val)

            # 初始化GTNNWR数据集
            print("📦 初始化GTNNWR数据集...")

            # 确定用于马氏距离计算的特征列
            if self.use_feature_mahalanobis and self.feature_columns_for_distance is None:
                # 默认使用所有特征列（排除空间和时间列）
                feature_columns_for_distance = self.gtnnwr_x_columns.copy()
                # 排除空间列
                if self.gtnnwr_spatial_columns:
                    feature_columns_for_distance = [col for col in feature_columns_for_distance
                                                    if col not in self.gtnnwr_spatial_columns]
                # 排除时间列
                if self.gtnnwr_temp_columns:
                    feature_columns_for_distance = [col for col in feature_columns_for_distance
                                                    if col not in self.gtnnwr_temp_columns]
                print(f"  📊 特征马氏距离: 使用 {len(feature_columns_for_distance)} 个特征")
            else:
                feature_columns_for_distance = self.feature_columns_for_distance

            try:
                # 使用init_dataset_split，传入特征马氏距离参数
                train_set, val_set, test_set = datasets.init_dataset_split(
                    train_data=train_data,
                    val_data=val_data,
                    test_data=val_data.head(max(1, min(5, len(val_data) // 2))),
                    x_column=self.gtnnwr_x_columns,
                    y_column=self.gtnnwr_y_column,
                    spatial_column=self.gtnnwr_spatial_columns,
                    temp_column=self.gtnnwr_temp_columns,
                    batch_size=min(1024, len(train_data)),
                    shuffle=False,
                    use_model="gtnnwr",
                    # 新增参数
                    use_feature_mahalanobis=self.use_feature_mahalanobis,
                    feature_columns_for_distance=feature_columns_for_distance
                )
                print(f"✅ 数据集初始化成功")
                print(f"  是否使用特征马氏距离: {self.use_feature_mahalanobis}")
                if self.use_feature_mahalanobis:
                    print(f"  马氏距离特征数: {len(feature_columns_for_distance)}")
            except Exception as error:
                print(f"❌ 数据集初始化失败: {error}")
                print("⚠️  跳过GTNNWR训练，返回None权重")
                return None, None

            print(f"✅ 数据集初始化完成:")
            print(f"  训练集样本数: {len(train_set) if hasattr(train_set, '__len__') else 'N/A'}")
            print(f"  验证集样本数: {len(val_set) if hasattr(val_set, '__len__') else 'N/A'}")

            # 检查数据集是否为空
            if (not hasattr(train_set, '__len__') or len(train_set) == 0 or
                    not hasattr(val_set, '__len__') or len(val_set) == 0):
                print(f"❌ 数据集为空或无效")
                print("⚠️  跳过GTNNWR训练，返回None权重")
                return None, None

            # 训练GTNNWR模型
            print("\n🏋️ 训练GTNNWR模型...")
            try:
                gtnnwr = models.GTNNWR(
                    train_dataset=train_set,
                    valid_dataset=val_set,
                    test_dataset=train_set,
                    dense_layers=self.gtnnwr_params.get('dense_layers', [[3], [512, 256, 64]]),
                    drop_out=self.gtnnwr_params.get('drop_out', 0.4),
                    optimizer=self.gtnnwr_params.get('optimizer', "Adadelta"),
                    optimizer_params=self.gtnnwr_params.get('optimizer_params', {}),
                    model_name=f"GTNNWR_Fold",
                    model_save_path="result/gtnnwr_models_temp",
                    log_path="result/gtnnwr_logs_temp",
                    write_path="result/gtnnwr_runs_temp"
                )

                # 添加图结构
                print("🕸️ 添加图结构...")
                gtnnwr.add_graph()

                # 训练
                print(f"⚙️ 训练参数: {self.gtnnwr_params.get('max_epoch', 3000)}轮, "
                      f"早停{self.gtnnwr_params.get('early_stop', 1000)}轮")

                gtnnwr.run(
                    max_epoch=self.gtnnwr_params.get('max_epoch', 3000),
                    early_stop=self.gtnnwr_params.get('early_stop', 1000),
                    print_frequency=self.gtnnwr_params.get('print_frequency', 100)
                )
            except Exception as model_error:
                print(f"❌ GTNNWR模型创建或训练失败: {model_error}")
                print("⚠️  跳过GTNNWR训练，返回None权重")
                return None, None

            # 提取权重矩阵
            def extract_weights(gtnnwr_instance, dataset, dataset_name="数据集"):
                """提取GTNNWR模型输出的权重矩阵"""
                if dataset is None or not hasattr(dataset, 'dataloader'):
                    print(f"  ❌ {dataset_name}无效或没有dataloader")
                    return None

                model = gtnnwr_instance._model
                model.eval()
                device = gtnnwr_instance._device

                all_weights = []
                sample_count = 0

                print(f"\n📥 从{dataset_name}提取权重...")

                with torch.no_grad():
                    try:
                        total_batches = 0
                        for batch_idx, batch in enumerate(dataset.dataloader):
                            if batch is None or len(batch) < 2:
                                continue

                            distances, features = batch[:2]
                            distances = distances.to(device)

                            # 获取模型输出
                            weights = model(distances)

                            # 检查权重中的NaN
                            if torch.isnan(weights).any():
                                print(f"  ⚠️  批次{batch_idx}权重中包含NaN值，使用1填充")
                                weights = torch.nan_to_num(weights, nan=1.0)

                            all_weights.append(weights.cpu().numpy())
                            sample_count += weights.shape[0]
                            total_batches += 1

                        print(f"  ✅ 完成: 总共处理{total_batches}个批次，{sample_count}个样本")

                    except Exception as e:
                        print(f"  ❌ 提取权重时出错: {e}")
                        import traceback
                        print(traceback.format_exc())
                        return None

                if all_weights:
                    weights_combined = np.concatenate(all_weights, axis=0)

                    # 检查并处理NaN值
                    nan_count = np.isnan(weights_combined).sum()
                    if nan_count > 0:
                        print(f"  ⚠️  权重矩阵中有{nan_count}个NaN值，使用1填充")
                        weights_combined = np.nan_to_num(weights_combined, nan=1.0)

                    print(f"  ✅ 提取完成: {weights_combined.shape} (样本数×特征数)")
                    return weights_combined
                else:
                    print(f"  ❌ 提取失败: 没有获取到权重")
                    return None

            # 提取训练集和验证集权重
            train_weights = extract_weights(gtnnwr, train_set, "训练集")
            val_weights = extract_weights(gtnnwr, val_set, "验证集")

            if train_weights is not None and val_weights is not None:
                # 检查并调整维度
                expected_cols = len(self.gtnnwr_x_columns)

                print(f"\n🔧 维度检查与调整:")
                print(f"  期望特征数: {expected_cols}")

                # 检查训练集权重维度
                if train_weights.shape[1] != expected_cols:
                    print(f"  ⚠️  训练权重维度不匹配: {train_weights.shape[1]} != {expected_cols}")
                    if train_weights.shape[1] == expected_cols + 1:
                        train_weights = train_weights[:, :expected_cols]
                        print(f"  ✅ 修复：去掉最后一列，新形状: {train_weights.shape}")
                    elif train_weights.shape[1] > expected_cols:
                        train_weights = train_weights[:, :expected_cols]
                        print(f"  ✅ 修复：截断到期望长度，新形状: {train_weights.shape}")
                    else:
                        padding = np.ones((train_weights.shape[0], expected_cols - train_weights.shape[1]))
                        train_weights = np.hstack([train_weights, padding])
                        print(f"  ✅ 修复：填充到期望长度，新形状: {train_weights.shape}")

                # 检查验证集权重维度
                if val_weights.shape[1] != expected_cols:
                    print(f"  ⚠️  验证权重维度不匹配: {val_weights.shape[1]} != {expected_cols}")
                    if val_weights.shape[1] == expected_cols + 1:
                        val_weights = val_weights[:, :expected_cols]
                        print(f"  ✅ 修复：去掉最后一列，新形状: {val_weights.shape}")
                    elif val_weights.shape[1] > expected_cols:
                        val_weights = val_weights[:, :expected_cols]
                        print(f"  ✅ 修复：截断到期望长度，新形状: {val_weights.shape}")
                    else:
                        padding = np.ones((val_weights.shape[0], expected_cols - val_weights.shape[1]))
                        val_weights = np.hstack([val_weights, padding])
                        print(f"  ✅ 修复：填充到期望长度，新形状: {val_weights.shape}")

                self.logger.debug(f"  提取到权重矩阵: 训练集{train_weights.shape}, 验证集{val_weights.shape}")
                return train_weights, val_weights
            else:
                print(f"\n❌ GTNNWR权重提取失败")
                self.logger.warning("  未能提取到权重矩阵")
                return None, None

        except Exception as e:
            print(f"\n❌ GTNNWR训练失败: {str(e)}")
            import traceback
            print(f"详细错误:\n{traceback.format_exc()}")
            self.logger.warning(f"  GTNNWR训练失败: {str(e)}")
            return None, None

    def run_complete_analysis(self, df, output_dir=None):
        """运行完整分析流程"""
        self.logger.info("=" * 70)
        self.logger.info("🚀 开始GTNNW-XGBoost完整分析流程")
        self.logger.info(f"使用特征马氏距离: {self.use_feature_mahalanobis}")
        self.logger.info("=" * 70)

        # 创建输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = f"./gtnnw_xgboost_results_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"输出目录: {output_dir}")

        try:
            # 1. 数据预处理
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 1: 数据预处理")
            self.logger.info("=" * 50)

            X, y, station_groups, year_groups, gtnnwr_data = self.preprocess_data(df, is_training=True)

            results = {
                'preprocessing': {
                    'samples': len(X),
                    'features': len(self.feature_columns),
                    'stations': len(np.unique(station_groups)),
                    'years': len(np.unique(year_groups)),
                    'use_gtnnwr': self.use_gtnnwr,
                    'use_feature_mahalanobis': self.use_feature_mahalanobis,
                    'nan_strategy': self.nan_strategy,
                    'nan_fill_stats': self.nan_fill_stats
                }
            }

            # 2. 年度交叉验证
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 2: 年度交叉验证")
            self.logger.info("=" * 50)

            results['yearly_cv'] = self.cross_validate(
                X, y, year_groups, 'yearly', gtnnwr_data
            )

            # 3. 站点交叉验证
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 3: 站点交叉验证")
            self.logger.info("=" * 50)

            results['station_cv'] = self.cross_validate(
                X, y, station_groups, 'station', gtnnwr_data
            )

            # 4. 训练最终模型
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 4: 训练最终模型")
            self.logger.info("=" * 50)

            results['final_model'] = self.train_final_model(X, y, gtnnwr_data)

            # 5. 特征重要性分析
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 5: 特征重要性分析")
            self.logger.info("=" * 50)

            results['feature_importance'] = self.get_feature_importance()

            # 6. 保存结果
            self.logger.info("\n" + "=" * 50)
            self.logger.info("步骤 6: 保存结果")
            self.logger.info("=" * 50)

            self._save_results(results, output_dir)

            # 7. 生成报告
            report = self._generate_report(results)
            print(report)

            self.logger.info("🎯 完整分析完成！")
            return results

        except Exception as e:
            self.logger.error(f"❌ 分析流程失败: {str(e)}")
            raise

    def _save_results(self, results, output_dir):
        """保存结果"""
        try:
            # 保存最终模型
            if 'final_model' in results:
                model_path = f'{output_dir}/final_model.pkl'
                joblib.dump(results['final_model'], model_path)
                self.logger.info(f"✅ 模型保存: {model_path}")

            # 保存NaN处理信息
            nan_info_path = f'{output_dir}/nan_handling_info.json'
            nan_info = {
                'strategy': self.nan_strategy,
                'fill_values': self.nan_fill_values,
                'fill_stats': self.nan_fill_stats
            }
            with open(nan_info_path, 'w', encoding='utf-8') as f:
                json.dump(nan_info, f, indent=2, ensure_ascii=False,
                          default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            self.logger.info(f"✅ NaN处理信息保存: {nan_info_path}")

            # 保存详细结果
            eval_results = {
                'training_info': {
                    'timestamp': datetime.now().isoformat(),
                    'feature_columns': self.feature_columns,
                    'gtnnwr_x_columns': self.gtnnwr_x_columns,
                    'use_gtnnwr': self.use_gtnnwr,
                    'use_feature_mahalanobis': self.use_feature_mahalanobis,
                    'nan_strategy': self.nan_strategy,
                    'total_samples': results.get('preprocessing', {}).get('samples', 0)
                },
                'model_parameters': self.params,
                'gtnnwr_parameters': self.gtnnwr_params,
                'station_cross_validation': results.get('station_cv', {}),
                'yearly_cross_validation': results.get('yearly_cv', {})
            }

            eval_path = f'{output_dir}/evaluation_results.json'
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False, default=float)
            self.logger.info(f"✅ 详细评估结果保存: {eval_path}")

            # 生成可视化
            self._create_scatter_plots(results, output_dir)

        except Exception as e:
            self.logger.error(f"保存结果失败: {str(e)}")

    def _generate_report(self, results):
        """生成分析报告"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("📊 GTNNW-XGBoost模型分析报告")
        report_lines.append("=" * 80)
        report_lines.append(f"使用GTNNWR权重增强: {self.use_gtnnwr}")
        report_lines.append(f"使用特征马氏距离: {self.use_feature_mahalanobis}")
        report_lines.append(f"NaN处理策略: {self.nan_strategy}")
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # 站点CV结果
        if 'station_cv' in results:
            station = results['station_cv']
            report_lines.append("📍 站点交叉验证 (空间评估):")
            report_lines.append(f"  聚合MAE: {station['overall']['MAE']:.3f} mm")
            report_lines.append(f"  聚合RMSE: {station['overall']['RMSE']:.3f} mm")
            report_lines.append(f"  聚合R: {station['overall']['R']:.3f}")
            report_lines.append(f"  折叠数: {station['folds']}")
            report_lines.append("")

        # 年度CV结果
        if 'yearly_cv' in results:
            yearly = results['yearly_cv']
            report_lines.append("📅 年度交叉验证 (时间评估):")
            report_lines.append(f"  聚合MAE: {yearly['overall']['MAE']:.3f} mm")
            report_lines.append(f"  聚合RMSE: {yearly['overall']['RMSE']:.3f} mm")
            report_lines.append(f"  聚合R: {yearly['overall']['R']:.3f}")
            report_lines.append(f"  折叠数: {yearly['folds']}")
            report_lines.append("")

        report_lines.append("\n" + "=" * 80)
        return "\n".join(report_lines)


# 便捷使用函数 - 新增支持特征马氏距离
def train_gtnnw_xgboost_model(data_df, output_dir=None, use_gtnnwr=True,
                              nan_strategy='median', nan_fill_value=0.0,
                              use_feature_mahalanobis=False,
                              feature_columns_for_distance=None):
    """便捷函数：训练GTNNW-XGBoost模型

    Args:
        data_df (pd.DataFrame): 包含特征和SWE的数据
        output_dir (str, optional): 输出目录路径
        use_gtnnwr (bool): 是否使用GTNNWR权重
        nan_strategy (str): NaN处理策略
        nan_fill_value (float): 填充NaN的值
        use_feature_mahalanobis (bool): 是否使用特征马氏距离
        feature_columns_for_distance (list): 用于马氏距离计算的特征列

    Returns:
        dict: 包含所有训练结果的字典
    """
    trainer = GTNNW_XGBoostTrainer(
        use_gtnnwr=use_gtnnwr,
        nan_strategy=nan_strategy,
        nan_fill_value=nan_fill_value,
        use_feature_mahalanobis=use_feature_mahalanobis,
        feature_columns_for_distance=feature_columns_for_distance
    )
    return trainer.run_complete_analysis(data_df, output_dir)


# 对比实验函数 - 新增支持特征马氏距离
def compare_models(data_df, output_dir=None):
    """对比不同配置的性能"""

    print("=" * 80)
    print("🔬 开始模型对比实验")
    print("=" * 80)

    # 1. 纯XGBoost
    print("\n1. 训练纯XGBoost模型...")
    xgb_trainer = GTNNW_XGBoostTrainer(use_gtnnwr=False, nan_strategy='median')
    xgb_results = xgb_trainer.run_complete_analysis(
        data_df,
        output_dir=os.path.join(output_dir, "xgboost_only") if output_dir else None
    )

    # 2. GTNNW-XGBoost (无特征马氏距离)
    print("\n2. 训练GTNNW-XGBoost模型 (无特征马氏距离)...")
    gtnnw_trainer1 = GTNNW_XGBoostTrainer(use_gtnnwr=True, nan_strategy='median',
                                          use_feature_mahalanobis=False)
    gtnnw_results1 = gtnnw_trainer1.run_complete_analysis(
        data_df,
        output_dir=os.path.join(output_dir, "gtnnw_xgboost_no_mahalanobis") if output_dir else None
    )

    # 3. GTNNW-XGBoost (有特征马氏距离)
    print("\n3. 训练GTNNW-XGBoost模型 (有特征马氏距离)...")
    gtnnw_trainer2 = GTNNW_XGBoostTrainer(use_gtnnwr=True, nan_strategy='median',
                                          use_feature_mahalanobis=True)
    gtnnw_results2 = gtnnw_trainer2.run_complete_analysis(
        data_df,
        output_dir=os.path.join(output_dir, "gtnnw_xgboost_with_mahalanobis") if output_dir else None
    )

    # 4. 对比分析
    print("\n" + "=" * 80)
    print("📊 模型对比结果")
    print("=" * 80)

    results_to_compare = [
        ("纯XGBoost", xgb_results),
        ("GTNNW-XGBoost (无马氏距离)", gtnnw_results1),
        ("GTNNW-XGBoost (有马氏距离)", gtnnw_results2)
    ]

    for name, res in results_to_compare:
        if 'station_cv' in res and 'overall' in res['station_cv']:
            r = res['station_cv']['overall']['R']
            if not np.isnan(r):
                print(f"{name}:")
                print(f"  站点CV R = {r:.3f}")
                print(f"  MAE = {res['station_cv']['overall']['MAE']:.3f} mm")
                print(f"  RMSE = {res['station_cv']['overall']['RMSE']:.3f} mm")
                print()

    return {
        'xgboost': xgb_results,
        'gtnnw_xgboost_no_mahalanobis': gtnnw_results1,
        'gtnnw_xgboost_with_mahalanobis': gtnnw_results2
    }