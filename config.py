import yaml
import os

# 定义常量
CONFIG_FILE = "config.yaml"
FOLDERS = ["model_save_dir", "log_dir", "results_dir", "visualization_dir"]
# 从 config.yaml 中读取所有支持的融合策略
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
FUSION_METHODS = ['weighted_voting', 'stacking', 'bagging', 'boosting', 'meta_learning', 'hierarchical']

def load_config():
    # 加载配置文件
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 创建必要的文件夹
    for dir_key in FOLDERS:
        dir_path = config.get(dir_key)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if dir_key in ['model_save_dir', 'log_dir', 'results_dir', 'visualization_dir']:
            for method in FUSION_METHODS:
                method_dir = os.path.join(dir_path, method)
                if not os.path.exists(method_dir):
                    os.makedirs(method_dir)
    
    return config