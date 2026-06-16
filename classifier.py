"""
机器学习分类器模块
实现SVM、随机森林、XGBoost、逻辑回归等分类器
"""

import numpy as np
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import xgboost as xgb
import pickle
import os

from config import CLASSIFIERS, RANDOM_SEED, CV_FOLDS


class TextSimilarityClassifier:
    """
    文本相似度分类器
    基于Embedding特征进行句子对相似度判断
    """
    
    def __init__(self, classifier_name, **kwargs):
        """
        初始化分类器
        
        Args:
            classifier_name: 分类器名称 ('svm', 'random_forest', 'xgboost', 'logistic_regression')
            **kwargs: 额外的模型参数
        """
        self.classifier_name = classifier_name
        self.model = self._build_model(classifier_name, **kwargs)
        self.training_time = 0
        self.is_trained = False
        
    def _build_model(self, classifier_name, **kwargs):
        """构建模型实例"""
        if classifier_name == "svm":
            return SVC(**kwargs)
        elif classifier_name == "random_forest":
            return RandomForestClassifier(**kwargs)
        elif classifier_name == "xgboost":
            return xgb.XGBClassifier(**kwargs)
        elif classifier_name == "logistic_regression":
            return LogisticRegression(**kwargs)
        else:
            raise ValueError(f"不支持的分类器: {classifier_name}")
    
    def train(self, X_train, y_train):
        """
        训练模型
        
        Args:
            X_train: 训练特征 (n_samples, n_features)
            y_train: 训练标签 (n_samples,)
        """
        print(f"[INFO] 训练 {self.classifier_name} 模型...")
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        print(f"[INFO] 训练完成，耗时: {self.training_time:.2f}秒")
        
    def predict(self, X):
        """预测类别"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练！")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练！")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # SVM需要probability=True才能输出概率
            return self.model.decision_function(X)
    
    def evaluate(self, X_test, y_test):
        """
        评估模型性能
        
        Returns:
            dict: 包含各项评估指标的字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练！")
        
        # 预测
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # 计算指标
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='binary'),
            "recall": recall_score(y_test, y_pred, average='binary'),
            "f1": f1_score(y_test, y_pred, average='binary'),
            "auc": roc_auc_score(y_test, y_proba),
            "training_time": self.training_time,
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        results["confusion_matrix"] = cm
        
        return results, y_pred, y_proba
    
    def cross_validate(self, X, y, cv=CV_FOLDS):
        """
        交叉验证
        
        Returns:
            dict: 各指标的交叉验证分数
        """
        print(f"[INFO] 进行{cv}折交叉验证...")
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
        
        scores = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            try:
                cv_scores = cross_val_score(self.model, X, y, cv=cv_splitter, scoring=metric)
                scores[f"cv_{metric}"] = {
                    "mean": cv_scores.mean(),
                    "std": cv_scores.std(),
                    "scores": cv_scores
                }
            except Exception as e:
                print(f"[WARNING] {metric} 交叉验证失败: {e}")
        
        return scores
    
    def save_model(self, filepath):
        """保存模型"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"[INFO] 模型已保存至: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"[INFO] 模型已从 {filepath} 加载")


def run_experiment(X_train, y_train, X_test, y_test, classifier_configs, embedding_name):
    """
    运行完整实验：训练多个分类器并对比结果
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        classifier_configs: 分类器配置字典
        embedding_name: Embedding模型名称（用于输出）
        
    Returns:
        list: 所有实验结果
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"使用 Embedding 模型: {embedding_name}")
    print(f"{'='*60}")
    
    for clf_name, clf_config in classifier_configs.items():
        print(f"\n{'-'*50}")
        print(f"分类器: {clf_name}")
        print(f"{'-'*50}")
        
        # 创建并训练模型
        classifier = TextSimilarityClassifier(
            clf_config["model"],
            **clf_config["params"]
        )
        classifier.train(X_train, y_train)
        
        # 评估
        metrics, y_pred, y_proba = classifier.evaluate(X_test, y_test)
        
        # 打印结果
        print(f"\n评估结果:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  训练时间:   {metrics['training_time']:.2f}秒")
        
        # 保存结果
        result = {
            "embedding": embedding_name,
            "classifier": clf_name,
            **metrics
        }
        results.append(result)
    
    return results


if __name__ == "__main__":
    # 测试分类器
    from sklearn.datasets import make_classification
    
    # 生成模拟数据
    X, y = make_classification(n_samples=1000, n_features=100, n_classes=2, random_state=42)
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]
    
    # 测试各个分类器
    for clf_name, clf_config in CLASSIFIERS.items():
        print(f"\n{'='*50}")
        print(f"测试分类器: {clf_name}")
        print(f"{'='*50}")
        
        clf = TextSimilarityClassifier(clf_config["model"], **clf_config["params"])
        clf.train(X_train, y_train)
        metrics, _, _ = clf.evaluate(X_test, y_test)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
