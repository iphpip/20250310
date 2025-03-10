import yaml
import os

# 定义常量
CONFIG_FILE = "config.yaml"
FOLDERS = ["model_save_dir", "log_dir", "results_dir", "visualization_dir"]

def load_config():
    # 加载配置文件
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 创建必要的文件夹
    for dir_key in FOLDERS:
        dir_path = config.get(dir_key)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    return config
