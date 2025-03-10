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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

logger = setup_logger()
config = load_config()

logger = None  # 添加全局变量

def main():
    global logger
    if logger is None:
        logger = setup_logger()
        
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
        model = train_model(model, X_resampled, y_resampled, X_test_processed, y_test_encoded)
        save_model(model, model_name)

        pred, proba = test_model(model.__class__, input_size, output_size, model_name, X_test_processed)
        if pred is not None and proba is not None:
            all_preds.append(pred)
            all_probas.append(proba)
            visualize_results(y_test_encoded, pred, label_encoder, model_name)

            # 修改为以字典形式接收返回值
            metrics = calculate_metrics(y_test_encoded, pred, proba, label_encoder)
            accuracy = metrics['accuracy']
            recall = metrics['recall']
            f1 = metrics['f1_score']
            auc = metrics['auc']
            log_loss_value = metrics['log_loss']

            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Log Loss: {log_loss_value:.4f}")

    # 训练元模型
    meta_input_size = len(all_preds)
    meta_model = train_meta_model(all_preds, y_test_encoded, meta_input_size)

    # 模型融合
    if all_preds:
        fusion_methods = ['weighted_voting', 'stacking', 'bagging', 'boosting', 'meta_learning']
        results = {
            "True Labels": label_encoder.inverse_transform(y_test_encoded)
        }

        for method in fusion_methods:
            if method == 'weighted_voting':
                weights = [0.3, 0.3, 0.4]  # 根据模型性能调整权重
                final_pred = ensemble_models(all_preds, y_true=y_test_encoded, label_encoder=label_encoder,
                                             fusion_method=method, weights=weights)
            elif method =='stacking':
                final_pred = ensemble_models(all_preds, y_true=y_test_encoded, label_encoder=label_encoder,
                                             fusion_method=method)
            elif method == 'bagging':
                final_pred = ensemble_models(all_preds, y_true=y_test_encoded, fusion_method=method)
            elif method == 'boosting':
                final_pred = ensemble_models(all_preds, y_true=y_test_encoded, fusion_method=method)
            elif method == 'meta_learning':
                final_pred, final_proba = test_meta_model(meta_model, all_preds)
            else:
                raise ValueError(f"不支持的融合方法: {method}")

            if method != 'meta_learning':
                final_proba = sum(all_probas) / len(all_probas)

            # 修改为以字典形式接收返回值
            metrics = calculate_metrics(y_test_encoded, final_pred, final_proba, label_encoder)
            accuracy = metrics['accuracy']
            recall = metrics['recall']
            f1 = metrics['f1_score']
            auc = metrics['auc']
            log_loss_value = metrics['log_loss']

            logger.info(f"{method.capitalize()} Ensemble - Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Log Loss: {log_loss_value:.4f}")

            results[f"{method.capitalize()} Ensemble Predicted Labels"] = label_encoder.inverse_transform(final_pred)

        # 保存实验结果
        for i, model_name in enumerate(model_names):
            results[f"{model_name} Predicted Labels"] = label_encoder.inverse_transform(all_preds[i])
        results_df = pd.DataFrame(results)
        results_path = os.path.join(config['results_dir'], "test_results.csv")
        results_df.to_csv(results_path, index=False)


if __name__ == "__main__":
    main()