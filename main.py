import warnings
warnings.filterwarnings("ignore", category=FutureWarning)               

from data_loader import load_and_preprocess_data
from data_resampler import resample_data
from model_builder import build_models
from trainer import train_model, train_meta_model
from tester import test_model, test_meta_model
from saver_visualizer import save_model, visualize_results
from model_ensemble import ensemble_models
from metrics_calculator import calculate_metrics
from logger_setup import setup_logger
import os
import pandas as pd
import numpy as np
from config import load_config

logger = setup_logger()
config = load_config()

def main():
    global logger
    if logger is None:
        logger = setup_logger()
    # 修改数据加载部分
    (X_train_processed, X_test_processed, 
     y_train_encoded, y_test_encoded, 
     label_encoder, attack_types) = load_and_preprocess_data()
    X_resampled, y_resampled = resample_data(X_train_processed, y_train_encoded, label_encoder)

    input_size = X_resampled.shape[1]
    output_size = len(np.unique(y_resampled))

    models = build_models(input_size, output_size)
    model_names = config['model_choices']

    all_preds = []
    all_probas = []

    for model, model_name in zip(models, model_names):
        model = train_model(model, X_resampled, y_resampled, X_test_processed, y_test_encoded, model_name)
        # 保存模型并记录保存路径和文件名
        model_path = os.path.join(config['model_save_dir'], f'{model_name}.pth')
        save_model(model, model_name)
        logger.info(f"{model_name} 模型已保存至 {model_path}")

        pred, proba = test_model(model.__class__, input_size, output_size, model_name, X_test_processed)
        if pred is not None and proba is not None:
            all_preds.append(pred)
            all_probas.append(proba)
            visualize_results(y_test_encoded, pred, label_encoder, model_name)
            logger.info(f"{model_name} 模型结果可视化完成。")

            # 修改此处，从字典中获取评估指标
            metrics = calculate_metrics(y_test_encoded, pred, proba, label_encoder)
            accuracy = metrics['accuracy']
            recall = metrics['recall']
            f1 = metrics['f1_score']
            auc = metrics['auc']
            log_loss_value = metrics['log_loss']
            log_loss_value = float(log_loss_value)
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Log Loss: {log_loss_value:.4f}")

    # 训练元模型
    meta_input_size = len(all_preds)
    meta_model = train_meta_model(all_preds, y_test_encoded, meta_input_size)

    # 模型融合
    if all_preds:
        # 修改模型融合调用
        final_pred = ensemble_models(
            all_preds,
            y_true=y_test_encoded,
            label_encoder=label_encoder,
            fusion_method=config['model_ensemble']['fusion_method'],
            attack_types=attack_types,
            dynamic_update=config['model_ensemble']['dynamic_update']
        )

        fusion_methods = ['weighted_voting', 'stacking', 'bagging', 'boosting', 'meta_learning', 'hierarchical']
        results = {
            "True Labels": label_encoder.inverse_transform(y_test_encoded)
        }

        for method in fusion_methods:
            # 初始化融合结果
            final_proba = None
            method_params = {
                'all_preds': all_preds,
                'y_true': y_test_encoded,
                'label_encoder': label_encoder,
                'fusion_method': method,
                'attack_types': attack_types,
                'dynamic_update': config['model_ensemble']['dynamic_update']
            }

            # 根据融合方法设置参数
            if method == 'weighted_voting':
                method_params['weights'] = [0.3, 0.3, 0.4]  # 权重参数
                final_pred = ensemble_models(**method_params)
            elif method == 'stacking':
                final_pred = ensemble_models(**method_params)
            elif method == 'bagging':
                final_pred = ensemble_models(**method_params)
            elif method == 'boosting':
                final_pred = ensemble_models(**method_params)
            elif method == 'meta_learning':
                final_pred, final_proba = test_meta_model(meta_model, all_preds)
            elif method == 'hierarchical':
                final_pred = ensemble_models(**method_params)
            else:
                raise ValueError(f"不支持的融合方法: {method}")

            # 统一处理概率计算
            final_proba = final_proba if final_proba is not None else sum(all_probas) / len(all_probas)

            # 统一指标计算和日志记录
            metrics = calculate_metrics(y_test_encoded, final_pred, final_proba, label_encoder)
            logger.info(
                f"{method.capitalize()} Ensemble - "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Recall: {metrics['recall']:.4f}, "
                f"F1: {metrics['f1_score']:.4f}, "
                f"AUC: {metrics['auc']:.4f}, "
                f"Log Loss: {metrics['log_loss']:.4f}"
            )
            
            results[f"{method.capitalize()} Ensemble Predicted Labels"] = label_encoder.inverse_transform(final_pred)
        # 保存实验结果
        for i, model_name in enumerate(model_names):
            results[f"{model_name} Predicted Labels"] = label_encoder.inverse_transform(all_preds[i])
        results_df = pd.DataFrame(results)
        results_path = os.path.join(config['results_dir'], "test_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"实验结果已保存至 {results_path}")

if __name__ == "__main__":
    main()