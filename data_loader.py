"""
数据加载与预处理模块
支持LCQMC、ATEC等中文语义相似度数据集
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config import DATA_DIR, RANDOM_SEED, TEST_SIZE, MAX_SAMPLES


def load_lcqmc_data(data_path=None):
    """
    加载LCQMC语义相似度数据集
    支持多种格式：.txt, .tsv, .csv
    如果本地没有，则生成增强版模拟数据用于测试
    
    数据格式：
    - sentence1: 句子1
    - sentence2: 句子2
    - label: 0(不相似) / 1(相似)
    """
    # 首先检查各种可能的数据文件
    possible_files = [
        os.path.join(DATA_DIR, 'train.tsv'),
        os.path.join(DATA_DIR, 'train.txt'),
        os.path.join(DATA_DIR, 'dev.csv'),
        os.path.join(DATA_DIR, 'dev.txt'),
        os.path.join(DATA_DIR, 'data.csv'),
    ]
    
    # 如果指定了路径，优先使用
    if data_path and os.path.exists(data_path):
        filepath = data_path
    else:
        # 自动查找可用的数据文件
        filepath = None
        for f in possible_files:
            if os.path.exists(f):
                filepath = f
                print(f"[INFO] 找到数据文件: {f}")
                break
    
    if filepath:
        # 根据文件扩展名选择分隔符
        if filepath.endswith('.csv'):
            sep = ','
        else:
            sep = '\t'
        
        try:
            # 尝试加载
            df = pd.read_csv(filepath, sep=sep, header=None, 
                            names=['sentence1', 'sentence2', 'label'])
            
            # 如果数据量超过限制，随机采样
            if len(df) > MAX_SAMPLES:
                print(f"[INFO] 数据量 {len(df)} 超过限制 {MAX_SAMPLES}，随机采样...")
                df = df.sample(n=MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
            
            print(f"[INFO] 成功加载数据: {len(df)} 条")
            print(f"[INFO] 数据样例:")
            print(df.head(3).to_string())
            return df
        except Exception as e:
            print(f"[WARNING] 加载失败: {e}")
    
    # 没有找到本地数据，生成增强版模拟数据
    print("[INFO] 未找到本地数据，生成增强版模拟数据...")
    from dataset_utils import generate_enhanced_mock_data
    return generate_enhanced_mock_data(n_samples=5000, output_file='enhanced_mock_data.csv')


def generate_mock_data(n_samples=2000):
    """
    生成模拟的语义相似度数据（用于快速验证代码）
    """
    np.random.seed(RANDOM_SEED)
    
    # 定义一些基础句子和它们的相似变体
    base_sentences = [
        ("怎么办理信用卡", "如何申请信用卡", 1),
        ("今天天气怎么样", "今天天气如何", 1),
        ("手机怎么充电", "手机如何充电", 1),
        ("怎么退款", "如何申请退款", 1),
        ("快递什么时候到", "包裹何时送达", 1),
        ("怎么修改密码", "如何更改密码", 1),
        ("怎么联系客服", "如何找客服", 1),
        ("商品有优惠吗", "有没有折扣", 1),
        ("怎么查看订单", "订单在哪里看", 1),
        ("怎么注销账号", "如何删除账户", 1),
    ]
    
    # 不相似的句子对
    dissimilar_pairs = [
        ("怎么办理信用卡", "今天天气怎么样", 0),
        ("手机怎么充电", "怎么退款", 0),
        ("快递什么时候到", "怎么修改密码", 0),
        ("怎么联系客服", "商品有优惠吗", 0),
        ("怎么查看订单", "怎么注销账号", 0),
        ("如何申请信用卡", "手机如何充电", 0),
        ("今天天气如何", "如何申请退款", 0),
        ("包裹何时送达", "如何更改密码", 0),
        ("如何找客服", "有没有折扣", 0),
        ("订单在哪里看", "如何删除账户", 0),
    ]
    
    data = []
    
    # 生成相似样本
    for _ in range(n_samples // 2):
        s1, s2, label = base_sentences[np.random.randint(len(base_sentences))]
        # 添加一些随机扰动
        data.append((s1, s2, label))
    
    # 生成不相似样本
    for _ in range(n_samples // 2):
        s1, s2, label = dissimilar_pairs[np.random.randint(len(dissimilar_pairs))]
        data.append((s1, s2, label))
    
    df = pd.DataFrame(data, columns=['sentence1', 'sentence2', 'label'])
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    return df


def prepare_data(df, test_size=TEST_SIZE):
    """
    准备训练集和测试集
    
    返回:
        X_train: 训练集句子对列表 [(s1, s2), ...]
        X_test: 测试集句子对列表
        y_train: 训练集标签
        y_test: 测试集标签
    """
    X = list(zip(df['sentence1'].values, df['sentence2'].values))
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"[INFO] 数据集划分完成:")
    print(f"       训练集: {len(X_train)} 条")
    print(f"       测试集: {len(X_test)} 条")
    print(f"       正样本比例: {np.mean(y):.2%}")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(df, filename="processed_data.csv"):
    """保存处理后的数据"""
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"[INFO] 数据已保存至: {filepath}")


if __name__ == "__main__":
    # 测试数据加载
    df = load_lcqmc_data()
    print(f"\n数据样例:")
    print(df.head(10))
    print(f"\n数据统计:")
    print(df['label'].value_counts())
    
    X_train, X_test, y_train, y_test = prepare_data(df)
