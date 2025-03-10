import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from logger_setup import setup_logger
from config import load_config

config = load_config()
logger = setup_logger()


def resample_data(X_train, y_train, label_encoder):
    class_counts = pd.Series(y_train).value_counts()
    majority_count = class_counts.max()
    balanced_count = int(majority_count * 0.6)
    logger.info("已统计各类别数量并确定平衡数量")

    # 修改少数类和多数类的划分阈值
    minority_classes = class_counts[class_counts < 50].index
    majority_classes = class_counts[class_counts >= 50].index

    X_minority = X_train[pd.Series(y_train).isin(minority_classes)]
    y_minority = y_train[pd.Series(y_train).isin(minority_classes)]

    X_minority_resampled = None
    y_minority_resampled = None

    for minority_class in minority_classes:
        class_samples = X_minority[pd.Series(y_train)[pd.Series(y_train).isin(minority_classes)] == minority_class]
        class_labels = y_minority[pd.Series(y_minority) == minority_class]
        num_samples = class_samples.shape[0]
        unique_classes = np.unique(class_labels)
        if len(unique_classes) == 1:
            logger.warning(f"少数类别 {minority_class} 只有一个类别，跳过过采样操作。")
            continue

        if num_samples < 6:
            n_neighbors = max(1, num_samples - 1)
        else:
            n_neighbors = 6
        try:
            borderline_smote = BorderlineSMOTE(random_state=42, n_neighbors=n_neighbors)
        except TypeError:
            logger.warning(f"当前BorderlineSMOTE版本可能不支持n_neighbors参数，尝试使用默认设置。")
            borderline_smote = BorderlineSMOTE(random_state=42)
        X_minority_class_resampled, y_minority_class_resampled = borderline_smote.fit_resample(
            class_samples, class_labels
        )
        if X_minority_resampled is None:
            X_minority_resampled = X_minority_class_resampled
            y_minority_resampled = y_minority_class_resampled
        else:
            X_minority_resampled = np.vstack([X_minority_resampled, X_minority_class_resampled])
            y_minority_resampled = np.hstack([y_minority_resampled, y_minority_class_resampled])
    logger.info("极少量类别过采样完成")

    # 暂时不进行过滤，直接使用 majority_classes
    valid_majority_classes = majority_classes

    X_majority = X_train[pd.Series(y_train).isin(valid_majority_classes)]
    y_majority = y_train[pd.Series(y_train).isin(valid_majority_classes)]

    if len(y_majority) == 0:
        logger.warning(f"未筛选出多数类样本，当前多数类标签: {valid_majority_classes}")

    sampling_strategy = {}
    for cls in valid_majority_classes:
        original_label = label_encoder.inverse_transform([cls])[0]
        original_count = class_counts[cls]
        target_count = min(balanced_count, original_count)
        sampling_strategy[label_encoder.transform([original_label])[0]] = target_count

    random_undersampler = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)

    if X_majority.shape[0] == 0:
        logger.warning("没有符合条件的多数类样本，跳过欠采样步骤。")
        X_majority_resampled = X_majority
        y_majority_resampled = y_majority
    else:
        X_majority_resampled, y_majority_resampled = random_undersampler.fit_resample(
            X_majority, y_majority
        )
        logger.info("数量相对较多但仍不平衡的类别欠采样完成")

    if X_minority_resampled is not None:
        X_resampled = np.vstack([X_minority_resampled, X_majority_resampled])
        y_resampled = np.hstack([y_minority_resampled, y_majority_resampled])
    else:
        X_resampled = X_majority_resampled
        y_resampled = y_majority_resampled
    logger.info("过采样和欠采样后的数据合并完成")

    # 再次检查重采样后标签的唯一性和范围
    unique_labels = np.unique(y_resampled)
    n_classes = len(label_encoder.classes_)
    if len(unique_labels) != n_classes or np.min(unique_labels) < 0 or np.max(unique_labels) >= n_classes:
        logger.error("重采样后标签出现问题，请检查重采样逻辑。")
        raise ValueError("重采样后标签出现问题，请检查重采样逻辑。")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    pd.Series(y_train).value_counts().plot(kind='bar')
    plt.title('Before Resampling')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    pd.Series(y_resampled).value_counts().plot(kind='bar')
    plt.title('After Resampling')
    plt.xlabel('Class')
    plt.ylabel('Count')

    plt.tight_layout()
    resampling_vis_dir = os.path.join(config['visualization_dir'], 'resampling')
    if not os.path.exists(resampling_vis_dir):
        os.makedirs(resampling_vis_dir)
    plt.savefig(os.path.join(resampling_vis_dir, 'resampling_distribution.png'))
    logger.info("重采样前后的数据分布可视化图已保存为 resampling_distribution.png")

    return X_resampled, y_resampled