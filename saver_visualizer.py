import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
from config import load_config
import os

config = load_config()

def save_model(model, model_name):
    model_path = os.path.join(config['model_save_dir'], f'{model_name}.pth')
    torch.save(model.state_dict(), model_path)

def visualize_results(y_true, y_pred, label_encoder, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    vis_path = os.path.join(config['visualization_dir'], f'{model_name}_confusion_matrix.png')
    if not os.path.exists(os.path.dirname(vis_path)):
        os.makedirs(os.path.dirname(vis_path))
    plt.savefig(vis_path)
    plt.close()

def plot_fusion_metrics(weights_history, attack_types):
    plt.figure(figsize=(12, 6))
    # 绘制动态权重变化曲线
    plt.subplot(1, 2, 1)
    for i, w in enumerate(weights_history.T):
        plt.plot(w, label=f'Model {i+1}')
    plt.title('Dynamic Weight Adjustment')
    plt.legend()

    # 绘制攻击类型分布
    plt.subplot(1, 2, 2)
    sns.countplot(x=attack_types)
    plt.title('Attack Type Distribution')
    plt.savefig(os.path.join(config['visualization_dir'], 'fusion_metrics.png'))
    plt.close()

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, model_name):
    epochs = range(1, len(train_losses) + 1)

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    vis_path = os.path.join(config['visualization_dir'], f'{model_name}_training_history.png')
    if not os.path.exists(os.path.dirname(vis_path)):
        os.makedirs(os.path.dirname(vis_path))
    plt.savefig(vis_path)
    plt.close()

def plot_feature_importance(feature_importances, feature_names, model_name):
    indices = np.argsort(feature_importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_names)), feature_importances[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')

    vis_path = os.path.join(config['visualization_dir'], f'{model_name}_feature_importance.png')
    if not os.path.exists(os.path.dirname(vis_path)):
        os.makedirs(os.path.dirname(vis_path))
    plt.savefig(vis_path)
    plt.close()

def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    vis_path = os.path.join(config['visualization_dir'], f'{model_name}_roc_curve.png')
    if not os.path.exists(os.path.dirname(vis_path)):
        os.makedirs(os.path.dirname(vis_path))
    plt.savefig(vis_path)
    plt.close()

def plot_ensemble_results(results, model_names):
    fusion_methods = list(results.keys())
    metrics = ['accuracy', 'recall', 'f1_score']
    num_metrics = len(metrics)
    num_methods = len(fusion_methods)

    bar_width = 0.2
    index = np.arange(num_methods)

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [results[method][metric] for method in fusion_methods]
        plt.bar(index + i * bar_width, values, bar_width, label=metric)

    plt.xlabel('Fusion Methods')
    plt.ylabel('Metrics')
    plt.title('Model Ensemble Results')
    plt.xticks(index + bar_width * (num_metrics - 1) / 2, fusion_methods)
    plt.legend()

    vis_path = os.path.join(config['visualization_dir'], 'ensemble_results.png')
    if not os.path.exists(os.path.dirname(vis_path)):
        os.makedirs(os.path.dirname(vis_path))
    plt.savefig(vis_path)
    plt.close()