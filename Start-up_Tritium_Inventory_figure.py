import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set(style='whitegrid')

# 定义保存路径
save_dir = 'D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp'
os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在，创建目录

# 读取 CSV 文件
df = pd.read_csv('D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp/combined_simulation_results.csv')

# 选择 sds.I 列（前 5 列，避免图表过于拥挤）
sds_columns = [col for col in df.columns if col.startswith('sds.I')][:20]

# 计算每列的第一行值减去最小值
differences = []
for col in sds_columns:
    min_value = df[col].min()
    first_row_value = df[col].iloc[0]
    difference = first_row_value - min_value
    differences.append(difference)

# 将计算结果保存到新的 CSV 文件
result_df = pd.DataFrame({
    'Parameter': sds_columns,
    'First_Minus_Min': differences
})
csv_path = os.path.join(save_dir, 'sds_I_differences.csv')
result_df.to_csv(csv_path, index=False)
print(f"计算结果已保存到 {csv_path}")

# 绘制折线图（只显示线）
plt.figure(figsize=(10, 6))
x_indices = range(len(sds_columns))  # 用索引作为 X 轴
plt.plot(x_indices, differences, color='skyblue', linewidth=2)

plt.xlabel('Combustion Parameters')
plt.ylabel('Start-up Tritium Inventory (g)')
plt.title('Difference Between First Row and Minimum of sds.I for Different Parameters')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.savefig('sds_I_first_minus_min.png', dpi=300, bbox_inches='tight')
plt.close()

print("figure save success!")