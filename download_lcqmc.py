"""
LCQMC数据集下载工具
尝试从多个镜像源下载数据集
"""

import os
import urllib.request
import zipfile
import pandas as pd
from config import DATA_DIR


def download_with_progress(url, save_path):
    """带进度条的下载"""
    print(f"[INFO] 正在下载: {url}")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r[INFO] 下载进度: {percent}%", end="")
    
    try:
        urllib.request.urlretrieve(url, save_path, reporthook)
        print(f"\n[INFO] 下载完成: {save_path}")
        return True
    except Exception as e:
        print(f"\n[ERROR] 下载失败: {e}")
        return False


def try_download_lcqmc():
    """
    尝试从多个源下载LCQMC数据集
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 备用下载链接
    sources = [
        {
            "name": "魔搭社区",
            "url": "https://modelscope.cn/datasets/C-MTEB/LCQMC",
            "type": "webpage"
        },
        {
            "name": "CSDN资源",
            "url": "https://download.csdn.net/download/weixin_42137700/10642786",
            "type": "direct"
        },
    ]
    
    print("="*60)
    print("LCQMC数据集下载")
    print("="*60)
    print()
    print("由于原官方链接失效，请从以下地址手动下载:")
    print()
    
    for i, source in enumerate(sources, 1):
        print(f"{i}. {source['name']}: {source['url']}")
    
    print()
    print("="*60)
    print("手动下载步骤:")
    print("="*60)
    print("1. 访问魔搭社区: https://modelscope.cn/datasets/C-MTEB/LCQMC")
    print("2. 注册/登录账号")
    print("3. 找到数据集下载链接")
    print("4. 下载 train.txt, dev.txt, test.txt")
    print(f"5. 将文件放入: {DATA_DIR}")
    print()
    print("数据格式应为:")
    print("  sentence1\\tsentence2\\tlabel")
    print("  怎么办理信用卡\\t如何申请信用卡\\t1")
    print("  今天天气怎么样\\t手机怎么充电\\t0")
    print()
    
    return False


def create_sample_data():
    """
    创建示例数据文件（用于测试格式）
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    sample_data = """怎么办理信用卡\t如何申请信用卡\t1
今天天气怎么样\t今天天气如何\t1
手机怎么充电\t手机如何充电\t1
怎么退款\t如何申请退款\t1
快递什么时候到\t包裹何时送达\t1
怎么修改密码\t如何更改密码\t1
怎么联系客服\t如何找客服\t1
商品有优惠吗\t有没有折扣\t1
怎么查看订单\t订单在哪里看\t1
怎么注销账号\t如何删除账户\t1
怎么办理信用卡\t今天天气怎么样\t0
手机怎么充电\t怎么退款\t0
快递什么时候到\t怎么修改密码\t0
怎么联系客服\t商品有优惠吗\t0
怎么查看订单\t怎么注销账号\t0"""
    
    sample_path = os.path.join(DATA_DIR, "sample_train.txt")
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(sample_data)
    
    print(f"[INFO] 示例数据已创建: {sample_path}")
    print("[INFO] 你可以参考这个格式准备自己的数据")
    
    return sample_path


if __name__ == "__main__":
    print("LCQMC数据集下载工具")
    print("="*60)
    
    # 检查是否已有数据
    data_files = ['train.txt', 'dev.txt', 'test.txt']
    existing = [f for f in data_files if os.path.exists(os.path.join(DATA_DIR, f))]
    
    if existing:
        print(f"[INFO] 发现已有数据文件: {existing}")
        print("[INFO] 如需重新下载，请删除这些文件")
    else:
        try_download_lcqmc()
        create_sample_data()
