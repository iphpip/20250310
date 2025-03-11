import logging
import os
from config import load_config

config = load_config()

def setup_logger(fusion_method):
    # 设置日志文件路径，包含融合策略子文件夹
    log_dir = os.path.join(config['log_dir'], fusion_method)
    log_file_path = os.path.join(log_dir, f"{fusion_method}_{config['log_file']}")
    
    # 配置日志记录器，包含融合策略信息
    logger = logging.getLogger(f"CyberSecurity_{fusion_method}")
    if logger.handlers:
        return logger
  
    # 设置日志级别
    log_level = getattr(logging, config['log_level'].upper())
    logger.setLevel(log_level) 
    
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
    
    # 初始化日志系统
    logger.info(f"初始化日志系统完成")
    logger.info(f"启用融合策略: {config['model_ensemble']['fusion_method']}")
    logger.info(f"动态更新状态: {config['model_ensemble']['dynamic_update']}")
    return logger