import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from config import load_config
from logger_setup import setup_logger
from sklearn.metrics import accuracy_score
from model_ensemble import MetaModel
import numpy as np

config = load_config()
logger = setup_logger()


def train_model(model, X_train, y_train, X_val, y_val, model_name):
    logger.info(f"开始训练 {model_name} 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=0.0001)

    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
    if hasattr(X_val, 'toarray'):
        X_val = X_val.toarray()

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config['train']['num_epochs']):
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            _, val_preds = torch.max(val_outputs, 1)
            val_acc = accuracy_score(y_val_tensor.cpu().numpy(), val_preds.cpu().numpy())

        logger.info(f'{model_name} - Epoch {epoch + 1}/{config["train"]["num_epochs"]}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    model.load_state_dict(best_model_state)
    logger.info(f"{model_name} 模型训练完成。")
    return model


def train_meta_model(all_preds, y_train, input_size):
    logger.info("开始训练元模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_model = MetaModel(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(meta_model.parameters(), lr=config['train']['learning_rate'])

    stacked_data = np.column_stack(all_preds)
    X_train_tensor = torch.FloatTensor(stacked_data).to(device)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(config['train']['num_epochs']):
        meta_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = meta_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        logger.info(f'Epoch {epoch + 1}/{config["train"]["num_epochs"]}, Meta Train Loss: {train_loss:.4f}')

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            best_model_state = meta_model.state_dict()

    meta_model.load_state_dict(best_model_state)
    logger.info("元模型训练完成。")
    return meta_model