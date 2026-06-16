"""
数据集下载工具
支持从多个来源下载LCQMC等中文语义相似度数据集
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from config import DATA_DIR


def download_from_url(url, save_path):
    """从URL下载文件"""
    print(f"[INFO] 正在下载: {url}")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"[INFO] 下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}")
        return False


def download_lcqmc_alternative():
    """
    尝试从多个备用源下载LCQMC数据集
    """
    # 备用下载链接
    urls = [
        # 百度网盘分享（需要提取码）
        # "https://pan.baidu.com/s/1...",
        
        # 阿里云盘或其他镜像
        # 由于原链接失效，这里提供手动下载指导
    ]
    
    print("[INFO] LCQMC数据集官方链接已失效")
    print("[INFO] 请从以下地址手动下载:")
    print("  1. 魔搭社区: https://modelscope.cn/datasets/C-MTEB/LCQMC")
    print("  2. 百度网盘搜索: LCQMC数据集")
    print("  3. CSDN资源: https://wenku.csdn.net/doc/7u56fm8219")
    print()
    print("[INFO] 下载后请将文件放入:")
    print(f"  {DATA_DIR}")
    print()
    print("[INFO] 数据格式应为:")
    print("  sentence1\\tsentence2\\tlabel")
    print("  怎么办理信用卡\\t如何申请信用卡\\t1")
    print("  今天天气怎么样\\t手机怎么充电\\t0")
    
    return False


def generate_enhanced_mock_data(n_samples=5000, output_file=None):
    """
    生成增强版模拟数据
    比原版更复杂、更多样化，更接近真实数据分布
    
    包含：
    - 更多样化的句子模板
    - 添加噪声和变体
    - 更复杂的相似/不相似判断
    """
    np.random.seed(42)
    
    # 定义更多样化的句子模板
    # 格式: (类别, [同义表达1, 同义表达2, ...])
    sentence_templates = {
        "信用卡": [
            "怎么办理信用卡", "如何申请信用卡", "信用卡怎么申请", 
            "办信用卡需要什么条件", "申请信用卡的流程是什么",
            "我想办一张信用卡", "信用卡办理流程",
        ],
        "天气": [
            "今天天气怎么样", "今天天气如何", "外面天气好吗",
            "今天会下雨吗", "天气情况如何", "今天气温多少度",
        ],
        "手机充电": [
            "手机怎么充电", "手机如何充电", "充电速度慢怎么办",
            "手机充不进电", "怎么让手机充电更快", "充电器怎么选",
        ],
        "退款": [
            "怎么退款", "如何申请退款", "退款流程是什么",
            "我想退货", "怎么办理退货", "退款多久到账",
        ],
        "快递": [
            "快递什么时候到", "包裹何时送达", "物流信息怎么查",
            "快递到哪里了", "怎么查快递进度", "预计什么时候送达",
        ],
        "密码": [
            "怎么修改密码", "如何更改密码", "密码忘记了怎么办",
            "怎么重置密码", "密码怎么设置", "修改登录密码",
        ],
        "客服": [
            "怎么联系客服", "如何找客服", "客服电话是多少",
            "怎么找人工客服", "在线客服在哪里", "客服怎么联系",
        ],
        "优惠": [
            "商品有优惠吗", "有没有折扣", "现在有什么活动",
            "怎么领取优惠券", "有满减活动吗", "会员有什么优惠",
        ],
        "订单": [
            "怎么查看订单", "订单在哪里看", "我的订单怎么查",
            "订单状态怎么查", "怎么跟踪订单", "订单详情怎么看",
        ],
        "账号": [
            "怎么注销账号", "如何删除账户", "账号怎么注销",
            "怎么退出登录", "怎么解绑手机号", "账号安全设置",
        ],
        "火车票": [
            "怎么买火车票", "如何购买火车票", "火车票怎么预订",
            "怎么查火车票", "火车票余票查询", "怎么退火车票",
        ],
        "酒店": [
            "怎么订酒店", "如何预订酒店", "酒店怎么选",
            "怎么查酒店价格", "酒店预订流程", "怎么取消酒店订单",
        ],
        "电影票": [
            "怎么买电影票", "如何购买电影票", "电影票怎么预订",
            "怎么选座位", "电影票多少钱", "怎么退电影票",
        ],
        "外卖": [
            "怎么点外卖", "如何订外卖", "外卖怎么下单",
            "怎么查外卖进度", "外卖多久送到", "怎么取消外卖订单",
        ],
        "医院": [
            "怎么挂号", "如何预约挂号", "医院挂号流程",
            "怎么查检查报告", "怎么找医生", "怎么缴费",
        ],
    }
    
    data = []
    
    # 生成相似样本（同类别内的句子对）
    n_similar = n_samples // 2
    categories = list(sentence_templates.keys())
    
    for _ in range(n_similar):
        # 随机选择一个类别
        cat = np.random.choice(categories)
        sentences = sentence_templates[cat]
        
        # 从该类别中随机选两个句子（可能相同或不同）
        s1 = np.random.choice(sentences)
        s2 = np.random.choice(sentences)
        
        # 如果相同，稍微修改一下
        if s1 == s2 and len(sentences) > 1:
            s2 = np.random.choice([s for s in sentences if s != s1])
        
        data.append((s1, s2, 1))
    
    # 生成不相似样本（不同类别的句子对）
    n_dissimilar = n_samples - n_similar
    
    for _ in range(n_dissimilar):
        # 随机选择两个不同类别
        cat1, cat2 = np.random.choice(categories, size=2, replace=False)
        s1 = np.random.choice(sentence_templates[cat1])
        s2 = np.random.choice(sentence_templates[cat2])
        
        data.append((s1, s2, 0))
    
    # 打乱数据
    df = pd.DataFrame(data, columns=['sentence1', 'sentence2', 'label'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存数据
    if output_file:
        output_path = os.path.join(DATA_DIR, output_file)
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"[INFO] 增强版模拟数据已保存: {output_path}")
        print(f"[INFO] 总样本数: {len(df)}")
        print(f"[INFO] 相似样本: {sum(df['label'] == 1)}")
        print(f"[INFO] 不相似样本: {sum(df['label'] == 0)}")
    
    return df


def load_or_generate_data(use_enhanced=True, n_samples=5000):
    """
    加载数据：优先使用本地数据，否则生成模拟数据
    
    Args:
        use_enhanced: 是否使用增强版模拟数据
        n_samples: 样本数量
    """
    # 检查是否有本地数据
    data_files = ['lcqmc_train.txt', 'train.txt', 'data.csv']
    
    for filename in data_files:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"[INFO] 找到本地数据: {filepath}")
            
            # 尝试读取
            try:
                if filepath.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_csv(filepath, sep='\t', header=None, 
                                   names=['sentence1', 'sentence2', 'label'])
                
                print(f"[INFO] 成功加载数据，共 {len(df)} 条")
                return df
            except Exception as e:
                print(f"[WARNING] 读取失败: {e}")
                continue
    
    # 没有本地数据，生成模拟数据
    print("[INFO] 未找到本地数据，生成模拟数据...")
    
    if use_enhanced:
        return generate_enhanced_mock_data(n_samples, 'enhanced_mock_data.csv')
    else:
        # 回退到原始简单数据
        from data_loader import generate_mock_data
        return generate_mock_data(n_samples)


if __name__ == "__main__":
    # 测试生成增强版数据
    df = generate_enhanced_mock_data(5000, 'enhanced_mock_data.csv')
    
    print("\n数据样例:")
    print(df.head(10))
    
    print("\n数据统计:")
    print(df['label'].value_counts())
    
    print("\n相似样本示例:")
    print(df[df['label'] == 1].head(5)[['sentence1', 'sentence2']].to_string())
    
    print("\n不相似样本示例:")
    print(df[df['label'] == 0].head(5)[['sentence1', 'sentence2']].to_string())
