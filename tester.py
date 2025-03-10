import torch
from model_builder import NeuralNetwork, CNN, LSTM
from config import load_config
from logger_setup import setup_logger
from model_ensemble import MetaModel
import numpy as np

config = load_config()
logger = setup_logger()


def test_model(model_class, input_size, output_size, model_name, X_test):
    # 记录开始测试的日志
    logger.info(f"开始测试 {model_name} 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(input_size, output_size).to(device)
    model_path = f'{config["model_save_dir"]}/{model_name}.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # 检查 X_test 是否为稀疏矩阵，如果是则转换为密集数组
        if hasattr(X_test, 'toarray'):
            X_test = X_test.toarray()
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, pred = torch.max(outputs, 1)
        pred = pred.cpu().numpy()
        proba = torch.nn.functional.softmax(outputs.cpu(), dim=1).numpy()
        logger.info(f"{model_name} 模型测试完成。")
        return pred, proba
    except FileNotFoundError:
        logger.error(f"模型文件 {model_path} 未找到，请检查。")
        return None, None


def test_meta_model(meta_model, all_preds):
    logger.info("开始测试元模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_model = meta_model.to(device)
    meta_model.eval()

    stacked_data = np.column_stack(all_preds)
    X_test_tensor = torch.FloatTensor(stacked_data).to(device)

    with torch.no_grad():
        outputs = meta_model(X_test_tensor)
        final_pred = (outputs > 0.5).float().cpu().numpy().flatten()
        proba = outputs.cpu().numpy().flatten()
    logger.info("元模型测试完成。")
    return final_pred, proba