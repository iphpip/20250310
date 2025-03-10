import numpy as np
from scipy.stats import entropy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import precision_score
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression

class MetaModel:
    """元学习模型包装器（新增）"""
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
def weighted_voting(all_preds, weights):
    """
    改进的加权投票融合策略（支持并行计算）
    :param all_preds: 各个模型的预测结果列表，每个元素是一个 numpy 数组
    :param weights: 各个模型的权重列表
    :return: 加权投票后的预测结果
    """
    num_samples = len(all_preds[0])
    final_pred = np.zeros(num_samples, dtype=int)
    
    # 并行计算加速
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(
            lambda i: np.argmax(
                np.sum([weights[j] * (pred[i] == cls) 
                       for j, pred in enumerate(all_preds)
                       for cls in np.unique(all_preds)]),
                axis=0
            ), range(num_samples))
        )
    
    final_pred = np.array(results)
    return final_pred

def dynamic_weight_adjustment(all_preds, y_true, decay_factor=0.9):
    """
    动态权重调整策略（新增）
    根据模型最新表现动态调整权重，增强对新型攻击的适应性
    """
    model_scores = [precision_score(y_true, pred) for pred in all_preds]
    weights = np.array(model_scores) / sum(model_scores)
    return weights * (decay_factor ** np.arange(len(weights))[::-1])

def hierarchical_fusion(all_preds, y_true, attack_types):
    """
    分层融合策略（新增）
    针对不同攻击类型进行分层融合，提升复杂攻击模式识别能力
    """
    type_groups = np.unique(attack_types)
    fused_preds = []
    
    for t in type_groups:
        mask = (attack_types == t)
        group_preds = [pred[mask] for pred in all_preds]
        fused = weighted_voting(group_preds, weights=[1]*len(group_preds))
        fused_preds.append((fused, mask))
    
    final_pred = np.empty_like(all_preds[0])
    for pred, mask in fused_preds:
        final_pred[mask] = pred
    return final_pred


def stacking(all_preds, y_true, label_encoder):
    """
    堆叠融合策略
    :param all_preds: 各个模型的预测结果列表，每个元素是一个 numpy 数组
    :param y_true: 真实标签
    :param label_encoder: 标签编码器
    :return: 堆叠融合后的预测结果
    """
    num_models = len(all_preds)
    stacked_data = np.column_stack(all_preds)
    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true)

    model = LogisticRegression()
    model.fit(stacked_data, y_true_encoded)

    num_samples = len(all_preds[0])
    test_stacked_data = stacked_data
    pred_encoded = model.predict(test_stacked_data)
    final_pred = label_encoder.inverse_transform(le.inverse_transform(pred_encoded))
    return final_pred


def bagging_fusion(all_preds, y_true):
    """
    Bagging 融合策略
    :param all_preds: 各个模型的预测结果列表，每个元素是一个 numpy 数组
    :param y_true: 真实标签
    :return: Bagging 融合后的预测结果
    """
    num_models = len(all_preds)
    stacked_data = np.column_stack(all_preds)

    # 使用 LogisticRegression 作为基分类器
    base_model = LogisticRegression()
    bagging_model = BaggingClassifier(base_estimator=base_model, n_estimators=num_models)
    bagging_model.fit(stacked_data, y_true)

    test_stacked_data = stacked_data
    final_pred = bagging_model.predict(test_stacked_data)
    return final_pred


def boosting_fusion(all_preds, y_true):
    """
    Boosting 融合策略
    :param all_preds: 各个模型的预测结果列表，每个元素是一个 numpy 数组
    :param y_true: 真实标签
    :return: Boosting 融合后的预测结果
    """
    num_models = len(all_preds)
    stacked_data = np.column_stack(all_preds)

    # 使用 LogisticRegression 作为基分类器
    base_model = LogisticRegression()
    boosting_model = AdaBoostClassifier(base_estimator=base_model, n_estimators=num_models)
    boosting_model.fit(stacked_data, y_true)

    test_stacked_data = stacked_data
    final_pred = boosting_model.predict(test_stacked_data)
    return final_pred


def ensemble_models(all_preds, y_true=None, label_encoder=None, fusion_method='weighted_voting', weights=None, attack_types=None, dynamic_update=False):
    """
    模型融合主函数
    :param all_preds: 各个模型的预测结果列表，每个元素是一个 numpy 数组
    :param y_true: 真实标签（仅在使用堆叠融合、Bagging 融合、Boosting 融合时需要）
    :param label_encoder: 标签编码器（仅在使用堆叠融合时需要）
    :param fusion_method: 融合方法，可选 'weighted_voting'、'stacking'、'bagging'、'boosting'、'hierarchical'
    :param weights: 加权投票时各个模型的权重列表
    :param attack_types: 攻击类型列表，用于分层融合
    :param dynamic_update: 是否动态更新权重
    :return: 融合后的预测结果
    """
    if dynamic_update:
        weights = dynamic_weight_adjustment(all_preds, y_true)
    
    if fusion_method == 'weighted_voting':
        if weights is None:
            weights = [1] * len(all_preds)
        return weighted_voting(all_preds, weights)
    elif fusion_method == 'stacking':
        if y_true is None or label_encoder is None:
            raise ValueError("使用堆叠融合时，y_true 和 label_encoder 不能为空")
        return stacking(all_preds, y_true, label_encoder)
    elif fusion_method == 'bagging':
        if y_true is None:
            raise ValueError("使用 Bagging 融合时，y_true 不能为空")
        return bagging_fusion(all_preds, y_true)
    elif fusion_method == 'boosting':
        if y_true is None:
            raise ValueError("使用 Boosting 融合时，y_true 不能为空")
        return boosting_fusion(all_preds, y_true)
    elif fusion_method == 'hierarchical':
        if y_true is None or attack_types is None:
            raise ValueError("使用分层融合时，y_true 和 attack_types 不能为空")
        return hierarchical_fusion(all_preds, y_true, attack_types)
    else:
        raise ValueError("不支持的融合方法，请选择 'weighted_voting'、'stacking'、'bagging'、'boosting'、'hierarchical'")
