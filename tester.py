import torch
from model_builder import NeuralNetwork, CNN, LSTM
from config import load_config
from logger_setup import setup_logger
from model_ensemble import MetaModel
import numpy as np

config = load_config()
logger = setup_logger()


def test_model(model_class, input_size, output_size, model_name, X_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(input_size, output_size).to(device)
    model_path = f'{config["model_save_dir"]}/{model_name}.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, pred = torch.max(outputs, 1)
        pred = pred.cpu().numpy()
        proba = torch.nn.functional.softmax(outputs.cpu(), dim=1).numpy()
        return pred, proba
    except FileNotFoundError:
        logger.error(f"模型文件 {model_path} 未找到，请检查。")
        return None, None


def test_meta_model(meta_model, all_preds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_model = meta_model.to(device)
    meta_model.eval()

    stacked_data = np.column_stack(all_preds)
    X_test_tensor = torch.FloatTensor(stacked_data).to(device)

    with torch.no_grad():
        outputs = meta_model(X_test_tensor)
        final_pred = (outputs > 0.5).float().cpu().numpy().flatten()
        proba = outputs.cpu().numpy().flatten()

    return final_pred, proba