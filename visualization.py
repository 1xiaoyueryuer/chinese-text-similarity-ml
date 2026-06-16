"""
可视化模块
生成实验结果图表，用于论文和PPT展示
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

# 尝试设置中文字体
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'PingFang SC', 'Heiti SC']
available_chinese_font = None
for font_name in chinese_fonts:
    try:
        fm.findfont(fm.FontProperties(family=font_name))
        available_chinese_font = font_name
        break
    except:
        continue

if available_chinese_font:
    plt.rcParams['font.sans-serif'] = [available_chinese_font, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    USE_CHINESE = True
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    USE_CHINESE = False
    print("[WARNING] 未找到中文字体，使用英文标签")

# 设置图表风格
sns.set_style("whitegrid")
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    pass


def plot_results(results_df, output_dir):
    """
    绘制实验结果对比图
    
    Args:
        results_df: 实验结果DataFrame
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 各指标柱状图对比
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # 按Embedding模型分组，绘制不同分类器的结果
        pivot_df = results_df.pivot(index='classifier', columns='embedding', values=metric)
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_title(f'{metric.upper()} 对比', fontsize=14, fontweight='bold')
        ax.set_xlabel('分类器', fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.legend(title='Embedding模型', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # 隐藏多余的子图
    if len(metrics) < len(axes):
        for idx in range(len(metrics), len(axes)):
            axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 指标对比图已保存")
    
    # 2. 热力图：Embedding模型 vs 分类器
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # F1-Score热力图
    f1_pivot = results_df.pivot(index='classifier', columns='embedding', values='f1')
    sns.heatmap(f1_pivot, annot=True, fmt='.4f', cmap='YlOrRd', 
                ax=axes[0], cbar_kws={'label': 'F1-Score'})
    axes[0].set_title('F1-Score 热力图', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Embedding模型', fontsize=12)
    axes[0].set_ylabel('分类器', fontsize=12)
    
    # AUC热力图
    auc_pivot = results_df.pivot(index='classifier', columns='embedding', values='auc')
    sns.heatmap(auc_pivot, annot=True, fmt='.4f', cmap='YlGnBu', 
                ax=axes[1], cbar_kws={'label': 'AUC'})
    axes[1].set_title('AUC 热力图', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Embedding模型', fontsize=12)
    axes[1].set_ylabel('分类器', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 热力图已保存")
    
    # 3. 训练时间对比
    fig, ax = plt.subplots(figsize=(12, 6))
    time_pivot = results_df.pivot(index='classifier', columns='embedding', values='training_time')
    time_pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('训练时间对比（秒）', fontsize=14, fontweight='bold')
    ax.set_xlabel('分类器', fontsize=12)
    ax.set_ylabel('训练时间 (秒)', fontsize=12)
    ax.legend(title='Embedding模型')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] 训练时间对比图已保存")


def plot_confusion_matrices(results_df, output_dir):
    """
    绘制混淆矩阵（需要保存混淆矩阵数据后调用）
    """
    # 这里简化处理，实际使用时可以从results_df中提取混淆矩阵
    pass


def plot_embedding_comparison(results_df, output_dir):
    """
    对比不同Embedding模型的整体性能
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算每个Embedding模型的平均性能
    embed_avg = results_df.groupby('embedding')[['accuracy', 'f1', 'auc']].mean()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(embed_avg.index))
    width = 0.25
    
    ax.bar(x - width, embed_avg['accuracy'], width, label='Accuracy', alpha=0.8)
    ax.bar(x, embed_avg['f1'], width, label='F1-Score', alpha=0.8)
    ax.bar(x + width, embed_avg['auc'], width, label='AUC', alpha=0.8)
    
    ax.set_xlabel('Embedding模型', fontsize=12)
    ax.set_ylabel('分数', fontsize=12)
    ax.set_title('不同Embedding模型平均性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(embed_avg.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Embedding模型对比图已保存")


def plot_tsne_visualization(embeddings, labels, output_dir, title="t-SNE可视化"):
    """
    使用t-SNE降维可视化Embedding向量
    
    Args:
        embeddings: 高维向量 (n_samples, n_features)
        labels: 类别标签 (n_samples,)
        output_dir: 输出目录
        title: 图表标题
    """
    print("[INFO] 正在进行t-SNE降维（可能需要一些时间）...")
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#e74c3c', '#3498db']
    labels_name = ['不相似', '相似']
    
    for i, (color, label_name) in enumerate(zip(colors, labels_name)):
        mask = labels == i
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                  c=color, label=label_name, alpha=0.6, s=30)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE维度1', fontsize=12)
    ax.set_ylabel('t-SNE维度2', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] t-SNE可视化图已保存")


def plot_roc_curves(results_list, y_test, output_dir):
    """
    绘制ROC曲线对比
    
    Args:
        results_list: 包含 (model_name, y_proba) 元组的列表
        y_test: 真实标签
        output_dir: 输出目录
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
    
    for (model_name, y_proba), color in zip(results_list, colors):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=color, lw=2, 
               label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC曲线对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] ROC曲线已保存")


def generate_summary_table(results_df, output_dir):
    """
    生成结果汇总表格（用于论文）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建汇总表
    summary = results_df.groupby(['embedding', 'classifier']).agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'f1': 'mean',
        'auc': 'mean',
        'training_time': 'mean'
    }).round(4)
    
    # 保存为CSV
    summary.to_csv(os.path.join(output_dir, 'summary_table.csv'), encoding='utf-8')
    
    # 保存为Markdown格式（方便插入论文）
    with open(os.path.join(output_dir, 'summary_table.md'), 'w', encoding='utf-8') as f:
        f.write("# 实验结果汇总\n\n")
        f.write(summary.to_markdown())
        f.write("\n\n")
        
        # 最佳结果
        best_idx = results_df['f1'].idxmax()
        best = results_df.loc[best_idx]
        f.write("## 最佳模型组合\n\n")
        f.write(f"- **Embedding模型**: {best['embedding']}\n")
        f.write(f"- **分类器**: {best['classifier']}\n")
        f.write(f"- **Accuracy**: {best['accuracy']:.4f}\n")
        f.write(f"- **F1-Score**: {best['f1']:.4f}\n")
        f.write(f"- **AUC**: {best['auc']:.4f}\n")
    
    print(f"[INFO] 汇总表格已保存")


if __name__ == "__main__":
    # 测试可视化
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成模拟结果数据
    np.random.seed(42)
    mock_results = []
    
    for embed in ['bge-small-zh', 'sbert-chinese', 'text2vec']:
        for clf in ['SVM', 'RandomForest', 'XGBoost', 'LogisticRegression']:
            mock_results.append({
                'embedding': embed,
                'classifier': clf,
                'accuracy': np.random.uniform(0.75, 0.95),
                'precision': np.random.uniform(0.75, 0.95),
                'recall': np.random.uniform(0.75, 0.95),
                'f1': np.random.uniform(0.75, 0.95),
                'auc': np.random.uniform(0.80, 0.98),
                'training_time': np.random.uniform(1, 60),
            })
    
    df = pd.DataFrame(mock_results)
    
    # 测试绘图
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_results(df, output_dir)
    plot_embedding_comparison(df, output_dir)
    generate_summary_table(df, output_dir)
    
    print("可视化测试完成！")
