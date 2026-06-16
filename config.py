"""
项目配置文件
包含模型参数、路径设置、实验配置等
"""

import os
import urllib.request

# 设置代理（根据你的梯子修改端口号）
# 当前梯子端口: 7897
PROXY_URL = "http://127.0.0.1:7897"
os.environ['HTTP_PROXY'] = PROXY_URL
os.environ['HTTPS_PROXY'] = PROXY_URL
os.environ['http_proxy'] = PROXY_URL
os.environ['https_proxy'] = PROXY_URL

# 为 urllib 设置代理（Hugging Face 下载器使用）
proxy_handler = urllib.request.ProxyHandler({'http': PROXY_URL, 'https': PROXY_URL})
opener = urllib.request.build_opener(proxy_handler)
urllib.request.install_opener(opener)

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 结果输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 模型缓存目录
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ==================== Embedding模型配置 ====================
# 要对比的Embedding模型列表（Hugging Face模型名）
EMBEDDING_MODELS = {
    "bge-small-zh": "BAAI/bge-small-zh",           # BGE中文轻量模型
    "bge-base-zh": "BAAI/bge-base-zh",             # BGE中文基础模型
    "sbert-chinese": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 多语言SBERT
    "text2vec": "shibing624/text2vec-base-chinese", # 中文text2vec
}

# Embedding维度（不同模型可能不同，这里统一取最大或按模型实际输出）
EMBEDDING_DIM = 768

# 最大序列长度
MAX_SEQ_LENGTH = 128

# 批次大小（根据你的显存调整）
BATCH_SIZE = 32

# 数据采样限制（为加快实验速度，只使用部分数据）
MAX_SAMPLES = 5000  # 最多使用 5000 条样本

# ==================== 机器学习模型配置 ====================
# 要对比的分类器
CLASSIFIERS = {
    "SVM": {
        "model": "svm",
        "params": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
        }
    },
    "RandomForest": {
        "model": "random_forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 20,
            "random_state": 42,
            "n_jobs": -1,
        }
    },
    "XGBoost": {
        "model": "xgboost",
        "params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1,
        }
    },
    "LogisticRegression": {
        "model": "logistic_regression",
        "params": {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        }
    },
}

# ==================== 实验配置 ====================
# 随机种子
RANDOM_SEED = 42

# 训练集/测试集划分比例
TEST_SIZE = 0.2

# 交叉验证折数
CV_FOLDS = 5

# ==================== 评估指标 ====================
METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
