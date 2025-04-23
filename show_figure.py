import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置绘图风格
sns.set(style='whitegrid')

# CSV 文件路径
csv_filename = "blanket.T_blanket.TBR.csv"  # 假设文件名，可后续通过参数传入
csv_path = os.path.join('D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp', csv_filename)

# 从文件名中解析参数名称
param_names = os.path.splitext(csv_filename)[0].split('_')
param_A_name = param_names[0]  # 例如 blanket.A
param_B_name = param_names[1]  # 例如 blanket.TBR

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 解析列名，提取参数 A 和参数 B 的值
param_columns = [col for col in df.columns if col != 'time']
param_A_values = {}
for col in param_columns:
    # 列名格式：paramA=valA_paramB=valB
    parts = col.split('_')
    param_A_part = parts[0]  # paramA=valA
    param_B_part = parts[1]  # paramB=valB
    param_A_val = float(param_A_part.split('=')[1])
    param_B_val = float(param_B_part.split('=')[1])
    if param_A_val not in param_A_values:
        param_A_values[param_A_val] = []
    param_A_values[param_A_val].append((param_B_val, col))

# 分组绘制：每张图显示最多 3 个参数 A 的值
param_A_list = sorted(param_A_values.keys())
group_size = 3  # 每组最多 3 个参数 A
for group_idx in range(0, len(param_A_list), group_size):
    group_A_values = param_A_list[group_idx:group_idx + group_size]
    
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('tab10', len(group_A_values))
    
    for i, param_A_val in enumerate(group_A_values):
        columns = param_A_values[param_A_val]
        for param_B_val, col in sorted(columns):  # 按参数 B 排序
            label = f"{param_A_name}={param_A_val:.3f}, {param_B_name}={param_B_val:.3f}"
            plt.plot(df['time'], df[col], label=label, color=colors[i], linewidth=1.5, alpha=0.8)

    plt.xlabel('Time (s)')
    plt.ylabel('sds.I')
    plt.title(f'Variation of sds.I with Time (Group {group_idx // group_size + 1})')
    plt.legend(loc='best', ncol=2, fontsize=8)
    plt.grid(True)
    plt.margins(y=0.1)

    # 保存图像
    save_dir = 'D:/FusionSimulationProgram/FFCAS_v0_FusionFuelCycleAnalysisSystem/temp'
    png_path = os.path.join(save_dir, f'sds_I_vs_time_{param_A_name}_{param_B_name}_group_{group_idx // group_size + 1}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"折线图已保存到 {png_path}")