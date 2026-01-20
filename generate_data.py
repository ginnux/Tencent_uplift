import pandas as pd
import numpy as np
import h5py
import os

# 1. 假设这是你的原始 DataFrame
# n_samples = 10000
# data = {
#     'X1': np.random.rand(n_samples),
#     'X2': np.random.rand(n_samples),
#     'T1': np.random.randint(0, 2, n_samples),
#     'Y': np.random.randint(0, 2, n_samples),
# }
# df = pd.DataFrame(data)

# 请替换成你自己的数据加载方式
df = pd.read_csv("fix.csv") 
# Y_COST	T_speed_fast	T_style_energetic	T_has_influencer	T_structure_aida	T_visual_closeup	T_overlay_subtitle	X_duration	X_shot_count	X_has_risk	X_min_age

# 2. 定义列顺序并创建 OUT_COLUMN_new 文件
# 确保这个顺序与你的 DataFrame 列名对应
# column_order = [f'X{i}' for i in range(1, 11)] + ['T1', 'Y'] # 示例：10个特征，1个处理，1个标签
# column_order = ["X_duration","X_shot_count", "X_has_risk", "X_min_age", "T_speed_fast", "T_style_energetic", "T_has_influencer", "T_structure_aida", "T_visual_closeup", "T_overlay_subtitle", "Y_COST"]
column_order = ["X_duration","X_shot_count", "X_has_risk", "X_min_age", "X_speed_fast", "X_style_energetic", "T_has_influencer", "X_structure_aida", "X_visual_closeup", "X_overlay_subtitle", "Y_COST"]
# column_order = list(df.columns) # 如果你的df已经是想要的顺序

# 创建目录和文件
os.makedirs("data/train_test_data", exist_ok=True)
with open("data/train_test_data/OUT_COLUMN_new", "w") as f:
    for col in column_order:
        f.write(col + '\n')

print("OUT_COLUMN_new 文件已创建。")

# 3. 按照指定顺序排列 DataFrame 并转换为 NumPy 数组
df_ordered = df[column_order]
data_array = df_ordered.to_numpy()

# 4. 将数据分成10份保存
n_splits = 10
split_size = len(data_array) // n_splits
for i in range(n_splits):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size if i < n_splits - 1 else len(data_array)
    split_data = data_array[start_idx:end_idx]
    
    # 保存为 HDF5 文件
    h5_filename = f"data/h5_data/train_part_{i+1}.h5"
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('data', data=split_data)
    
    print(f"保存了 {h5_filename}，包含 {len(split_data)} 条记录。")

# 创建保存HDF5文件的目录
os.makedirs("data/h5_data", exist_ok=True)
