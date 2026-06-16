# 基于Embedding模型的文本语义相似度分类研究

## 项目简介

本项目是机器学习课程报告，研究如何使用预训练的Embedding模型（Hugging Face）提取文本特征，结合传统机器学习分类器（SVM、随机森林、XGBoost等）进行文本语义相似度判断。

## 项目结构

```
.
├── config.py                   # 配置文件（模型参数、路径等）
├── data_loader.py              # 数据加载与预处理
├── embedding_extractor.py      # Embedding特征提取（Hugging Face模型）
├── classifier.py               # 机器学习分类器
├── visualization.py            # 可视化与结果分析
├── main.py                     # 主程序入口
├── requirements.txt            # Python依赖
├── README.md                   # 项目说明
├── data/                       # 数据目录
└── results/                    # 实验结果目录
```

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 硬件要求

- **CPU**: 支持（较慢）
- **GPU**: 推荐，可大幅加速Embedding提取
- **内存**: 建议8GB以上
- **硬盘**: 至少5GB（用于下载预训练模型）

## 快速开始

### 方式1：快速演示（推荐首次运行）

使用模拟数据快速验证代码：

```bash
python main.py --mode demo
```

### 方式2：完整实验

```bash
python main.py --mode full
```

如果需要重新提取特征（不使用缓存）：

```bash
python main.py --mode full --no-cache
```

## 实验流程

### 1. 数据准备

- 支持LCQMC、ATEC等中文语义相似度数据集
- 如果没有本地数据，会自动生成模拟数据

### 2. 特征提取

使用以下Embedding模型（Hugging Face）：

| 模型 | 说明 | 参数量 |
|:---|:---|:---|
| BAAI/bge-small-zh | BGE中文轻量模型 | 小 |
| BAAI/bge-base-zh | BGE中文基础模型 | 中 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 多语言SBERT | 小 |
| shibing624/text2vec-base-chinese | 中文Text2Vec | 中 |

### 3. 机器学习分类

对比以下分类器：

- **SVM**（支持向量机）
- **RandomForest**（随机森林）
- **XGBoost**（梯度提升树）
- **LogisticRegression**（逻辑回归）

### 4. 评估指标

- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-Score（F1分数）
- AUC（ROC曲线下面积）

## 实验结果

运行完成后，结果保存在 `results/` 目录：

- `experiment_results.csv` - 详细实验结果
- `metrics_comparison.png` - 各指标对比图
- `heatmap_comparison.png` - F1和AUC热力图
- `embedding_model_comparison.png` - Embedding模型对比
- `training_time_comparison.png` - 训练时间对比

## 论文写作参考

### 核心创新点

1. **多模型对比**：系统对比了4种主流Embedding模型在中文语义相似度任务上的表现
2. **特征融合策略**：采用 |u-v|, u*v, (u+v)/2 三种特征组合方式
3. **端到端流程**：从文本预处理到模型评估的完整机器学习流程

### 关键图表

1. 不同Embedding模型的性能对比柱状图
2. Embedding模型 × 分类器的性能热力图
3. t-SNE降维可视化（展示Embedding的区分能力）
4. ROC曲线对比

## 常见问题

### Q1: 下载模型很慢？

可以设置镜像加速：
```bash
# Windows PowerShell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python main.py
```

### Q2: 显存不足？

修改 `config.py` 中的 `BATCH_SIZE` 为更小的值（如8或16）

### Q3: 如何添加自己的数据？

将数据文件放入 `data/` 目录，格式为：
```
sentence1\tsentence2\tlabel
怎么办理信用卡\t如何申请信用卡\t1
今天天气怎么样\t手机怎么充电\t0
```

## 引用

```bibtex
@inproceedings{xiao2021cpm,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Xiao, Zhen and Zhang, Yuxiao and others},
  booktitle={AI Open},
  year={2021}
}

@article{reimers2019sentence,
  title={Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks},
  author={Reimers, Nils and Gurevych, Iryna},
  journal={EMNLP},
  year={2019}
}
```

## AI使用说明

本项目在开发过程中使用了以下AI工具：

- **ChatGPT**: 辅助代码设计、论文框架构思（约30%）
- **GitHub Copilot**: 代码补全与调试辅助（约20%）
- **人工完成**: 代码审查、实验分析、论文撰写（约50%）

## 许可证

MIT License

## 联系方式

如有问题，请通过课程群联系。
