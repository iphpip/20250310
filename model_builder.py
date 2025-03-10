import torch
import torch.nn as nn
from config import load_config
from logger_setup import setup_logger

config = load_config()
logger = setup_logger()


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, config['train']['hidden_size1'])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(config['train']['hidden_size1'], config['train']['hidden_size2'])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(config['train']['hidden_size2'], output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        # 深度可分离卷积层 1
        self.depthwise_conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, groups=1)
        self.pointwise_conv1 = nn.Conv1d(1, 16, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        # 深度可分离卷积层 2
        self.depthwise_conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1, groups=16)
        self.pointwise_conv2 = nn.Conv1d(16, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 深度可分离卷积层 3
        self.depthwise_conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=1, groups=32)
        self.pointwise_conv3 = nn.Conv1d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.fc1 = nn.Linear(64 * (input_size // 8), 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)

        # 第一层深度可分离卷积
        x = self.depthwise_conv1(x)
        x = self.pointwise_conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu1(x)
        x = self.pool1(x)

        # 第二层深度可分离卷积
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu2(x)
        x = self.pool2(x)

        # 第三层深度可分离卷积
        x = self.depthwise_conv3(x)
        x = self.pointwise_conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu3(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.leaky_relu4(x)
        x = self.dropout(x)

        x = self.fc2(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM, self).__init__()
        # 使用 LSTM 替代简单 RNN
        self.lstm = nn.LSTM(input_size, config['train']['hidden_size1'], num_layers=2, batch_first=True, dropout=0.2)
        self.bn = nn.BatchNorm1d(config['train']['hidden_size1'])
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc1 = nn.Linear(config['train']['hidden_size1'], 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.leaky_relu(out)
        out = self.fc1(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        return out


def build_models(input_size, output_size):
    model_classes = {
        "NeuralNetwork": NeuralNetwork,
        "CNN": CNN,
        "LSTM": LSTM
    }
    models = []
    for model_name in config['model_choices']:
        if model_name not in model_classes:
            logger.error(f"未找到模型类 {model_name}，请检查配置文件。")
            continue
        model = model_classes[model_name](input_size, output_size)
        models.append(model)
    return models