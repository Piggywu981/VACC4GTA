import pandas as pd
import numpy as np
import os
from sklearn.utils import resample

# 配置参数
RAW_DATA_PATH = '../data/raw/training_data.csv'
BALANCED_DATA_PATH = '../data/processed/balanced_training_data.csv'
MAX_SAMPLES_PER_CLASS = 5000  # 每个类别的最大样本数

# 确保输出目录存在
os.makedirs(os.path.dirname(BALANCED_DATA_PATH), exist_ok=True)

def load_raw_data(csv_path):
    """加载原始训练数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"原始数据文件不存在: {csv_path}")
    return pd.read_csv(csv_path)

def balance_data(df):
    """平衡数据集中各类别的样本数量"""
    # 分离不同类别的数据
    left_samples = df[df['left'] == 1]
    forward_samples = df[df['forward'] == 1]
    right_samples = df[df['right'] == 1]

    print(f"原始数据分布 - 左转: {len(left_samples)}, 直行: {len(forward_samples)}, 右转: {len(right_samples)}")

    # 确定每个类别的目标样本数（取最小值或设定的最大值）
    target_samples = min(MAX_SAMPLES_PER_CLASS, len(left_samples), len(forward_samples), len(right_samples))

    # 对每个类别进行重采样
    left_balanced = resample(left_samples, replace=False, n_samples=target_samples, random_state=42)
    forward_balanced = resample(forward_samples, replace=False, n_samples=target_samples, random_state=42)
    right_balanced = resample(right_samples, replace=False, n_samples=target_samples, random_state=42)

    # 合并平衡后的数据
    balanced_df = pd.concat([left_balanced, forward_balanced, right_balanced])
    # 打乱顺序
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"平衡后数据分布 - 总计: {len(balanced_df)} 样本")
    print(f"平衡后类别分布 - 左转: {len(balanced_df[balanced_df['left'] == 1])}, 直行: {len(balanced_df[balanced_df['forward'] == 1])}, 右转: {len(balanced_df[balanced_df['right'] == 1])}")

    return balanced_df

def main():
    try:
        # 加载原始数据
        raw_df = load_raw_data(RAW_DATA_PATH)
        print(f"成功加载原始数据: {len(raw_df)} 样本")

        # 平衡数据
        balanced_df = balance_data(raw_df)

        # 保存平衡后的数据
        balanced_df.to_csv(BALANCED_DATA_PATH, index=False)
        print(f"平衡后的数据已保存至: {BALANCED_DATA_PATH}")

    except Exception as e:
        print(f"数据处理出错: {e}")

if __name__ == '__main__':
    main()