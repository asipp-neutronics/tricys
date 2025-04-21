import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style='whitegrid')

# 读取 CSV 文件
df = pd.read_csv('D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp/combined_simulation_results.csv')

# 提取 time 列和数 个 sds.I 列（可以根据需要调整列数）
time = df['time']
sds_columns = [col for col in df.columns if col.startswith('sds.I')][:20:2]  # 列

# 创建折线图
plt.figure(figsize=(10, 6))
for col in sds_columns:
    plt.plot(time, df[col], label=col, linewidth=1.5)

# 设置标签和标题
plt.xlabel('Time (h)')
plt.ylabel('Tritium Inventory (g)')
plt.title('Variation of Tritium Inventory with Time under Different Burning Fractions')
plt.legend(loc='best')
plt.grid(True)

# 保存图像
plt.savefig('sds_I_vs_time.png', dpi=300, bbox_inches='tight')

print("figure save success!")