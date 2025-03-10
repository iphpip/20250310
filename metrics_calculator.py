from scipy.stats import entropy  # 添加缺失的entropy导入
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, log_loss
import numpy as np

def calculate_metrics(y_true, y_pred, y_proba, label_encoder, all_preds=None):
    # 初始化metrics字典
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'auc': roc_auc_score(
            np.eye(len(label_encoder.classes_))[y_true],  # 保持原有y_test_proba生成方式
            y_proba / np.sum(y_proba, axis=1, keepdims=True),
            multi_class='ovr', 
            average='micro'
        ),
        'log_loss': log_loss(y_true, y_proba)
    }

    # 新增模型多样性指标
    if all_preds is not None:
        metrics['ensemble_diversity'] = np.mean([entropy(np.bincount(pred)) for pred in all_preds])
    
    return metrics  # 移除重复的return语句