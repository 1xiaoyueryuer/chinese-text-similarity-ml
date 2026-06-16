"""
主程序：文本语义相似度分类实验
流程：数据加载 → Embedding特征提取 → 机器学习分类 → 结果对比
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from config import (
    EMBEDDING_MODELS, 
    CLASSIFIERS, 
    OUTPUT_DIR, 
    DATA_DIR
)
from data_loader import load_lcqmc_data, prepare_data
from embedding_extractor import extract_and_save_features, EmbeddingExtractor
from classifier import run_experiment
from visualization import plot_results, plot_confusion_matrices, plot_embedding_comparison


def run_full_experiment(use_cached_features=True):
    """
    运行完整实验流程
    
    Args:
        use_cached_features: 是否使用缓存的特征（如果存在）
    """
    print("="*70)
    print("文本语义相似度分类实验")
    print("基于Embedding模型 + 机器学习分类器")
    print("="*70)
    
    # ========== 1. 数据加载 ==========
    print("\n" + "="*70)
    print("步骤 1: 加载数据")
    print("="*70)
    
    # 尝试加载本地LCQMC数据，如果没有则使用模拟数据
    data_path = os.path.join(DATA_DIR, "lcqmc_train.txt")
    df = load_lcqmc_data(data_path if os.path.exists(data_path) else None)
    
    # 保存数据
    df.to_csv(os.path.join(DATA_DIR, "data.csv"), index=False, encoding='utf-8')
    
    # 划分数据集
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # 保存标签
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    
    # ========== 2. 特征提取（Embedding） ==========
    print("\n" + "="*70)
    print("步骤 2: 提取Embedding特征")
    print("="*70)
    
    all_results = []
    
    for embed_name, embed_model in EMBEDDING_MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# Embedding模型: {embed_name}")
        print(f"# 模型路径: {embed_model}")
        print(f"{'#'*60}")
        
        # 检查是否有缓存的特征
        feature_dir = os.path.join(OUTPUT_DIR, "features")
        train_feature_path = os.path.join(feature_dir, f"{embed_name}_train.npy")
        test_feature_path = os.path.join(feature_dir, f"{embed_name}_test.npy")
        
        # 检查特征维度是否匹配当前数据
        feature_valid = False
        if use_cached_features and os.path.exists(train_feature_path) and os.path.exists(test_feature_path):
            cached_train = np.load(train_feature_path)
            cached_test = np.load(test_feature_path)
            if cached_train.shape[0] == len(X_train) and cached_test.shape[0] == len(X_test):
                feature_valid = True
                print(f"[INFO] 发现缓存的特征且维度匹配，直接加载...")
                X_train_features = cached_train
                X_test_features = cached_test
            else:
                print(f"[INFO] 缓存特征维度不匹配 (train: {cached_train.shape[0]} vs {len(X_train)}, test: {cached_test.shape[0]} vs {len(X_test)})，重新提取...")
        
        if not feature_valid:
            # 提取特征
            X_train_features, X_test_features = extract_and_save_features(
                X_train, X_test, embed_name, feature_dir
            )
        
        # ========== 3. 机器学习分类 ==========
        print(f"\n{'='*70}")
        print("步骤 3: 训练并评估分类器")
        print(f"{'='*70}")
        
        results = run_experiment(
            X_train_features, y_train,
            X_test_features, y_test,
            CLASSIFIERS,
            embed_name
        )
        
        all_results.extend(results)
    
    # ========== 4. 结果汇总与可视化 ==========
    print("\n" + "="*70)
    print("步骤 4: 结果汇总与可视化")
    print("="*70)
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 保存结果
    results_path = os.path.join(OUTPUT_DIR, "experiment_results.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    print(f"\n[INFO] 实验结果已保存至: {results_path}")
    
    # 打印汇总表格
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    
    summary = results_df.groupby(['embedding', 'classifier']).agg({
        'accuracy': 'mean',
        'f1': 'mean',
        'auc': 'mean',
        'training_time': 'mean'
    }).round(4)
    
    print(summary.to_string())
    
    # 找出最佳组合
    best_idx = results_df['f1'].idxmax()
    best_result = results_df.loc[best_idx]
    
    print("\n" + "="*70)
    print("最佳模型组合")
    print("="*70)
    print(f"Embedding模型: {best_result['embedding']}")
    print(f"分类器: {best_result['classifier']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")
    print(f"F1-Score: {best_result['f1']:.4f}")
    print(f"AUC: {best_result['auc']:.4f}")
    
    # 生成可视化
    print("\n[INFO] 生成可视化图表...")
    try:
        plot_results(results_df, OUTPUT_DIR)
        plot_embedding_comparison(results_df, OUTPUT_DIR)
        print(f"[INFO] 图表已保存至: {OUTPUT_DIR}")
    except Exception as e:
        print(f"[WARNING] 可视化生成失败: {e}")
    
    # 保存详细结果
    detailed_results = {
        "best_model": {
            "embedding": str(best_result['embedding']),
            "classifier": str(best_result['classifier']),
            "accuracy": float(best_result['accuracy']),
            "f1": float(best_result['f1']),
            "auc": float(best_result['auc']),
        },
        "all_results": json.loads(results_df.to_json(orient='records'))
    }
    
    with open(os.path.join(OUTPUT_DIR, "detailed_results.json"), 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*70)
    print("实验完成！")
    print(f"所有结果保存在: {OUTPUT_DIR}")
    print("="*70)
    
    return results_df


def quick_demo():
    """
    快速演示：使用模拟数据和轻量级模型快速跑通流程
    适合第一次运行测试
    """
    print("="*70)
    print("快速演示模式（使用模拟数据）")
    print("="*70)
    
    # 使用模拟数据
    df = load_lcqmc_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # 只使用一个轻量级模型
    embed_name = "bge-small-zh"
    print(f"\n使用模型: {embed_name}")
    
    extractor = EmbeddingExtractor(embed_name)
    
    print("\n提取训练集特征...")
    X_train_features = extractor.encode_sentence_pairs(X_train[:100])  # 只取前100条演示
    print("提取测试集特征...")
    X_test_features = extractor.encode_sentence_pairs(X_test[:50])
    
    y_train_sub = y_train[:100]
    y_test_sub = y_test[:50]
    
    # 只训练一个分类器
    from classifier import TextSimilarityClassifier
    
    print("\n训练SVM分类器...")
    clf = TextSimilarityClassifier("svm", kernel='rbf', C=1.0, probability=True)
    clf.train(X_train_features, y_train_sub)
    
    metrics, _, _ = clf.evaluate(X_test_features, y_test_sub)
    
    print("\n评估结果:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    
    print("\n演示完成！代码运行正常。")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='文本语义相似度分类实验')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'demo'],
                       help='运行模式: full=完整实验, demo=快速演示')
    parser.add_argument('--no-cache', action='store_true',
                       help='不使用缓存的特征，重新提取')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        quick_demo()
    else:
        run_full_experiment(use_cached_features=not args.no_cache)
