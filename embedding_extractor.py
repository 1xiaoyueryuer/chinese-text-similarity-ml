"""
Embedding特征提取模块
使用Hugging Face的预训练模型将文本转换为向量表示
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

from config import (
    EMBEDDING_MODELS, 
    MAX_SEQ_LENGTH, 
    BATCH_SIZE, 
    MODEL_CACHE_DIR
)


class EmbeddingExtractor:
    """
    Embedding特征提取器
    支持多种预训练模型：BGE、Sentence-BERT、Text2Vec等
    """
    
    def __init__(self, model_name, device=None):
        """
        初始化Embedding提取器
        
        Args:
            model_name: Hugging Face模型名称或config中定义的简称
            device: 计算设备 ('cuda' 或 'cpu')
        """
        # 如果传入的是简称，转换为完整模型名
        if model_name in EMBEDDING_MODELS:
            self.model_name = EMBEDDING_MODELS[model_name]
            self.model_short_name = model_name
        else:
            self.model_name = model_name
            self.model_short_name = model_name.split('/')[-1]
        
        # 设置计算设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] 使用设备: {self.device}")
        
        # 加载模型和分词器
        print(f"[INFO] 正在加载模型: {self.model_name}")
        self._load_model()
        
    def _load_model(self):
        """加载预训练模型"""
        try:
            # 尝试使用sentence-transformers加载（更简洁的API）
            # 对BGE模型需要trust_remote_code
            trust = 'bge' in self.model_name.lower()
            self.model = SentenceTransformer(
                self.model_name, 
                cache_folder=MODEL_CACHE_DIR,
                trust_remote_code=trust
            )
            self.model.to(self.device)
            self.use_sentence_transformer = True
            print(f"[INFO] 使用SentenceTransformer加载成功")
        except Exception as e:
            print(f"[INFO] SentenceTransformer加载失败，使用transformers库: {e}")
            # 使用transformers库加载
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=MODEL_CACHE_DIR,
                trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()
            self.use_sentence_transformer = False
    
    def encode(self, texts, batch_size=BATCH_SIZE, show_progress=True):
        """
        将文本列表编码为向量
        
        Args:
            texts: 文本列表（字符串列表）
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        Returns:
            numpy数组，形状为 (len(texts), embedding_dim)
        """
        if self.use_sentence_transformer:
            # 使用SentenceTransformer的encode方法
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                device=self.device
            )
            return embeddings
        else:
            # 使用transformers库手动编码
            return self._encode_with_transformers(texts, batch_size, show_progress)
    
    def _encode_with_transformers(self, texts, batch_size, show_progress):
        """使用transformers库编码文本"""
        embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding")
        
        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i+batch_size]
                
                # 分词
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LENGTH,
                    return_tensors="pt"
                ).to(self.device)
                
                # 获取模型输出
                outputs = self.model(**inputs)
                
                # 使用[CLS] token的向量作为句子表示
                # 或者使用mean pooling
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def encode_sentence_pairs(self, sentence_pairs, batch_size=BATCH_SIZE):
        """
        编码句子对，返回拼接后的特征向量
        
        Args:
            sentence_pairs: 句子对列表 [(s1, s2), ...]
            
        Returns:
            numpy数组，形状为 (len(pairs), embedding_dim * 3)
            包含：|u-v|, u*v, (u+v)/2 三种特征组合
        """
        # 提取所有句子
        sentences1 = [pair[0] for pair in sentence_pairs]
        sentences2 = [pair[1] for pair in sentence_pairs]
        
        print(f"[INFO] 编码句子对，共 {len(sentence_pairs)} 对...")
        
        # 分别编码
        embeddings1 = self.encode(sentences1, batch_size=batch_size)
        embeddings2 = self.encode(sentences2, batch_size=batch_size)
        
        # 构建特征向量（多种组合方式）
        # 方法1: 直接拼接 [u; v]
        # features = np.hstack([embeddings1, embeddings2])
        
        # 方法2: 使用 |u-v|, u*v, (u+v)/2 （论文中常用的方法）
        diff = np.abs(embeddings1 - embeddings2)           # 元素差绝对值
        product = embeddings1 * embeddings2                 # 元素乘积
        avg = (embeddings1 + embeddings2) / 2               # 平均值
        
        features = np.hstack([diff, product, avg])
        
        print(f"[INFO] 特征维度: {features.shape[1]}")
        
        return features
    
    def get_embedding_dim(self):
        """获取Embedding维度"""
        test_text = ["测试"]
        test_embedding = self.encode(test_text, show_progress=False)
        return test_embedding.shape[1]


def extract_and_save_features(X_train, X_test, model_name, output_dir):
    """
    提取特征并保存
    
    Args:
        X_train: 训练集句子对
        X_test: 测试集句子对
        model_name: Embedding模型名称
        output_dir: 输出目录
        
    Returns:
        train_features, test_features
    """
    extractor = EmbeddingExtractor(model_name)
    
    # 提取训练集特征
    print(f"\n{'='*50}")
    print(f"提取训练集特征...")
    train_features = extractor.encode_sentence_pairs(X_train)
    
    # 提取测试集特征
    print(f"\n提取测试集特征...")
    test_features = extractor.encode_sentence_pairs(X_test)
    
    # 保存特征
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, f"{model_name}_train.npy")
    test_path = os.path.join(output_dir, f"{model_name}_test.npy")
    
    np.save(train_path, train_features)
    np.save(test_path, test_features)
    
    print(f"\n[INFO] 特征已保存:")
    print(f"       训练集: {train_path}, 形状: {train_features.shape}")
    print(f"       测试集: {test_path}, 形状: {test_features.shape}")
    
    return train_features, test_features


if __name__ == "__main__":
    # 测试Embedding提取器
    test_sentences = [
        "怎么办理信用卡",
        "如何申请信用卡",
        "今天天气怎么样",
    ]
    
    # 测试BGE模型
    print("="*60)
    print("测试 BGE-small-zh 模型")
    print("="*60)
    extractor = EmbeddingExtractor("bge-small-zh")
    embeddings = extractor.encode(test_sentences)
    print(f"Embedding形状: {embeddings.shape}")
    print(f"Embedding样例 (前5维):\n{embeddings[0, :5]}")
    
    # 测试句子对编码
    pairs = [("怎么办理信用卡", "如何申请信用卡")]
    pair_features = extractor.encode_sentence_pairs(pairs)
    print(f"\n句子对特征形状: {pair_features.shape}")
