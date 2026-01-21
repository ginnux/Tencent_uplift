import pandas as pd

df = pd.read_csv('fix.csv')

# 对Y_COST列进行二值化处理，按前75%分位数为0，其余为1
threshold = df['Y_COST'].quantile(0.75)
df['Y_COST_BIN'] = (df['Y_COST'] >= threshold).astype(int)

# 对Y_ROI同样操作
threshold = df['Y_ROI'].quantile(0.75)
df['Y_ROI_BIN'] = (df['Y_ROI'] >= threshold).astype(int)

# 保存处理后的数据到新的CSV文件
df.to_csv('fix_bin.csv', index=False)
print("Binarization complete. Output saved to 'fix_bin.csv'.")