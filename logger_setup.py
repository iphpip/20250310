import logging
import os
from config import load_config

config = load_config()

# 移除模块级别的直接日志调用
# 改为在setup_logger函数中添加日志处理器后记录

def setup_logger():
    # 设置日志文件路径
    log_file_path = os.path.join(config['log_dir'], config['log_file'])
    
    # 配置日志记录器
    #logger = logging.getLogger(__name__)
    #logger.setLevel(logging.INFO)
    logger = logging.getLogger("CyberSecurity")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO) 
    # 创建日志处理器
    file_handler = logging.FileHandler(log_file_path)
    stream_handler = logging.StreamHandler()
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # 添加处理器后记录配置
    logger.info(f"初始化日志系统完成")
    # 在日志输出中增加融合参数记录
    logger.info(f"启用融合策略: {config['model_ensemble']['fusion_method']}")
    logger.info(f"动态更新状态: {config['model_ensemble']['dynamic_update']}")
    return logger