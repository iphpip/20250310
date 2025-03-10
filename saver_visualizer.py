import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
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
    
    # 绘制攻击类型分布
    plt.subplot(1, 2, 2)
    sns.countplot(x=attack_types)
    plt.title('Attack Type Distribution')
    plt.savefig(os.path.join(config['visualization_dir'], 'fusion_metrics.png'))