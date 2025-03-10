import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import joblib
from logger_setup import setup_logger
from config import load_config

config = load_config()
logger = setup_logger()


def create_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    logger.info("已识别数值型和类别型特征")

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    logger.info("特征处理流程整合完成")
    return preprocessor


def load_and_preprocess_data(save_processed_data=False):
    file_path = config['dataset']['train']
    try:
        logger.info("开始读取数据")
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"文件 {file_path} 未找到，请检查文件路径。")
        raise
    except Exception as e:
        logger.error(f"读取数据时发生错误: {e}")
        raise
    data['attack_cat'] = data['attack_cat'].replace('Normal', 'normal')
    logger.info("数据读取完成，已统一标签类别")

    X = data.drop(['attack_cat', 'label'], axis=1)
    y = data['attack_cat']
    logger.info("特征和标签分离完成")

    class_counts = data['attack_cat'].value_counts()
    valid_classes = class_counts.nlargest(19).index
    valid_indices = y.isin(valid_classes)
    X = X[valid_indices]
    y = y[valid_indices]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info("数据集划分完成，训练集和测试集已生成")

    preprocessor = create_preprocessor(X)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.info("训练集和测试集特征预处理完成")

    label_encoder = LabelEncoder()
    all_labels = pd.concat([y_train, y_test]).unique()
    label_encoder.fit(all_labels)
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    logger.info("标签编码完成")

    if save_processed_data:
        processed_data_dir = os.path.join(config['results_dir'], 'processed_data')
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        joblib.dump(X_train_processed, os.path.join(processed_data_dir, 'X_train_processed.joblib'))
        joblib.dump(X_test_processed, os.path.join(processed_data_dir, 'X_test_processed.joblib'))
        joblib.dump(y_train_encoded, os.path.join(processed_data_dir, 'y_train_encoded.joblib'))
        joblib.dump(y_test_encoded, os.path.join(processed_data_dir, 'y_test_encoded.joblib'))
        logger.info("预处理后的数据已保存")

    # 新增攻击类型提取
    attack_types = data[config['model_ensemble']['attack_type_column']].values
    return (X_train_processed, X_test_processed, 
            y_train_encoded, y_test_encoded, 
            label_encoder, attack_types)  # 修改返回值为六元组