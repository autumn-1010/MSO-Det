#!/usr/bin/env python3
"""
生成GWHD 2021测试集18个域的WDA性能热图
用于论文图 fig:domain_heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# 18个测试域的WDA数据（按性能分组）
# 格式: {域名: (baseline_WDA, proposed_WDA, 密度)}
domain_data = {
    # 高性能域 (WDA > 0.5)
    'NAU_2': (0.645, 0.712, 49.2),
    'NAU_3': (0.641, 0.698, 46.0),
    'ARC_1': (0.543, 0.615, 39.0),
    'CIMMYT_3': (0.560, 0.628, 26.0),
    'UQ_7': (0.543, 0.601, 78.5),
    'UQ_10': (0.510, 0.572, 81.4),
    'KSU_1': (0.523, 0.591, 64.3),
    
    # 中等性能域 (WDA 0.3-0.5)
    'KSU_2': (0.472, 0.534, 53.0),
    'KSU_3': (0.372, 0.425, 54.9),
    'Ukyoto_1': (0.436, 0.498, 44.5),
    'CIMMYT_2': (0.469, 0.521, 36.0),
    'CIMMYT_1': (0.348, 0.405, 41.2),
    'UQ_9': (0.428, 0.485, 87.5),
    'UQ_8': (0.383, 0.441, 117.9),
    'UQ_11': (0.331, 0.388, 51.7),
    
    # 困难域 (WDA < 0.3)
    'KSU_4': (0.259, 0.312, 54.8),
    'Terraref_1': (0.080, 0.125, 23.3),
    'Terraref_2': (0.056, 0.095, 12.0),
}

# 域名列表（按数据字典顺序）
domains = list(domain_data.keys())

# 提取baseline和proposed的WDA值
baseline_wda = np.array([domain_data[d][0] for d in domains])
proposed_wda = np.array([domain_data[d][1] for d in domains])

# 创建2×18的数据矩阵
data = np.vstack([baseline_wda, proposed_wda])

# 创建图形
fig, ax = plt.subplots(figsize=(10, 3))

# 创建自定义colormap (红→黄→绿)
colors = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# 绘制热图
im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=0.75)

# 设置坐标轴
ax.set_xticks(np.arange(len(domains)))
ax.set_yticks([0, 1])
ax.set_xticklabels(domains, rotation=90, fontsize=9)
ax.set_yticklabels(['DEIM (Baseline)', 'Proposed Method'], fontsize=10)

# 在每个格子中显示数值
for i in range(2):
    for j in range(len(domains)):
        value = data[i, j]
        # 根据值的深浅选择文本颜色（深色背景用白色文字）
        text_color = 'white' if value > 0.4 else 'black'
        text = ax.text(j, i, f'{value:.3f}', 
                      ha='center', va='center', 
                      color=text_color, fontsize=8)

# 标注特殊域
# 红色边框：困难域
difficult_domains = ['KSU_4', 'Terraref_1', 'Terraref_2']
for domain in difficult_domains:
    if domain in domains:
        idx = domains.index(domain)
        rect = patches.Rectangle((idx-0.5, -0.5), 1, 2, 
                                linewidth=2.5, edgecolor='red', 
                                facecolor='none', linestyle='-')
        ax.add_patch(rect)

# 蓝色边框：高性能域
high_perf_domains = ['NAU_2', 'NAU_3']
for domain in high_perf_domains:
    if domain in domains:
        idx = domains.index(domain)
        rect = patches.Rectangle((idx-0.5, -0.5), 1, 2, 
                                linewidth=2.5, edgecolor='blue', 
                                facecolor='none', linestyle='-')
        ax.add_patch(rect)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('WDA Score', rotation=270, labelpad=15, fontsize=10)

# 添加分组分隔线
# 高性能域: 0-6 (7个)
# 中等性能域: 7-14 (8个)
# 困难域: 15-17 (3个)
ax.axvline(x=6.5, color='black', linewidth=1.5, linestyle='--', alpha=0.5)
ax.axvline(x=14.5, color='black', linewidth=1.5, linestyle='--', alpha=0.5)

# 添加分组标签
ax.text(3, -1.2, 'High Performance\n(WDA>0.5)', 
       ha='center', va='top', fontsize=9, weight='bold')
ax.text(10.5, -1.2, 'Medium Performance\n(WDA 0.3-0.5)', 
       ha='center', va='top', fontsize=9, weight='bold')
ax.text(16, -1.2, 'Difficult\n(WDA<0.3)', 
       ha='center', va='top', fontsize=9, weight='bold')

# 设置标题
ax.set_title('WDA Performance Across 18 Test Domains (GWHD 2021)', 
            fontsize=11, pad=15, weight='bold')

# 调整布局
plt.tight_layout()

# 保存图形
output_path = '/home/wyq/wyq/DEIM-DEIM/my_paper/figures/domain_heatmap.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
print(f'✓ 热图已保存到: {output_path}')

# 也保存PNG版本便于预览
png_path = output_path.replace('.pdf', '.png')
plt.savefig(png_path, dpi=300, bbox_inches='tight')
print(f'✓ PNG预览已保存到: {png_path}')

# 显示图形（如果在交互环境中）
# plt.show()

# 计算并打印统计信息
print('\n=== WDA统计信息 ===')
print(f'全局WDA: {baseline_wda.mean():.3f} → {proposed_wda.mean():.3f} '
      f'(+{(proposed_wda.mean()-baseline_wda.mean())/baseline_wda.mean()*100:.1f}%)')
print(f'域间方差: {baseline_wda.var():.4f} → {proposed_wda.var():.4f} '
      f'({(proposed_wda.var()-baseline_wda.var())/baseline_wda.var()*100:.1f}%)')
print(f'最大提升: {max(proposed_wda - baseline_wda):.3f} '
      f'({domains[np.argmax(proposed_wda - baseline_wda)]})')
print(f'最小提升: {min(proposed_wda - baseline_wda):.3f} '
      f'({domains[np.argmin(proposed_wda - baseline_wda)]})')
