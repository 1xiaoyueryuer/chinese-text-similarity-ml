"""
数据集自动下载工具
尝试从多个源下载LCQMC数据集
"""

import os
import urllib.request
import pandas as pd
from config import DATA_DIR


def download_from_aistudio():
    """
    从飞桨AI Studio下载LCQMC数据集
    地址: https://aistudio.baidu.com/datasetdetail/146584
    """
    print("="*60)
    print("LCQMC数据集下载")
    print("="*60)
    print()
    print("由于原官方链接失效，推荐从以下渠道获取:")
    print()
    print("1. 飞桨AI Studio（推荐）:")
    print("   https://aistudio.baidu.com/datasetdetail/146584")
    print("   - 注册百度账号")
    print("   - 进入页面点击'下载'")
    print()
    print("2. 魔搭社区:")
    print("   https://modelscope.cn/datasets/C-MTEB/LCQMC")
    print("   - 注册阿里云账号")
    print("   - 搜索LCQMC数据集")
    print()
    print("3. CSDN资源:")
    print("   https://wenku.csdn.net/doc/7u56fm8219")
    print("   - 搜索LCQMC数据集下载")
    print()
    print("="*60)
    print("数据格式说明")
    print("="*60)
    print("下载后应有3个文件:")
    print("  - train.txt (约24万条)")
    print("  - dev.txt (约8800条)")
    print("  - test.txt (约12500条)")
    print()
    print("文件格式（制表符分隔）:")
    print("  sentence1\\tsentence2\\tlabel")
    print("  怎么办理信用卡\\t如何申请信用卡\\t1")
    print("  今天天气怎么样\\t手机怎么充电\\t0")
    print()
    print(f"请将文件放入: {DATA_DIR}")
    print()
    
    return False


def check_existing_data():
    """检查是否已有数据文件"""
    data_files = {
        'train.txt': '训练集',
        'dev.txt': '验证集',
        'test.txt': '测试集'
    }
    
    found = []
    for filename, desc in data_files.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"[INFO] 发现 {desc}: {filename} ({size/1024/1024:.2f} MB)")
            found.append(filename)
    
    return found


def load_data_file(filepath):
    """加载数据文件"""
    print(f"[INFO] 正在加载: {filepath}")
    
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                data.append({
                    'sentence1': parts[0],
                    'sentence2': parts[1],
                    'label': int(parts[2])
                })
    
    df = pd.DataFrame(data)
    print(f"[INFO] 成功加载 {len(df)} 条数据")
    print(f"[INFO] 相似样本: {sum(df['label'] == 1)}")
    print(f"[INFO] 不相似样本: {sum(df['label'] == 0)}")
    
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("LCQMC数据集检查工具")
    print("="*60)
    
    # 检查已有数据
    existing = check_existing_data()
    
    if existing:
        print(f"\n[INFO] 已找到 {len(existing)} 个数据文件")
        print("[INFO] 可以直接运行实验: python main.py --mode full")
    else:
        print("\n[INFO] 未找到数据文件")
        download_from_aistudio()
        
        # 创建示例文件
        print("\n[INFO] 创建示例数据文件供参考...")
        sample_path = os.path.join(DATA_DIR, "sample_format.txt")
        with open(sample_path, 'w', encoding='utf-8') as f:
            f.write("怎么办理信用卡\t如何申请信用卡\t1\n")
            f.write("今天天气怎么样\t今天天气如何\t1\n")
            f.write("手机怎么充电\t手机如何充电\t1\n")
            f.write("怎么退款\t如何申请退款\t1\n")
            f.write("快递什么时候到\t包裹何时送达\t1\n")
            f.write("怎么办理信用卡\t今天天气怎么样\t0\n")
            f.write("手机怎么充电\t怎么退款\t0\n")
            f.write("快递什么时候到\t怎么修改密码\t0\n")
        
        print(f"[INFO] 示例文件已创建: {sample_path}")
