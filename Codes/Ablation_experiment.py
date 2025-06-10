import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体和样式
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.axisbelow'] = True

# 消融实验数据
models = [
    'Baseline\n(ResNet50)',
    'Baseline\n+SA',
    'Baseline\n+SA+CA',
    'Baseline\n+SA+CA+PE',
    'DermaTrans\n(Full Model)'
]

# 根据评估指标修改的性能数据
metrics = {
    'Accuracy': [0.871, 0.878, 0.883, 0.887, 0.892],
    'Precision': [0.868, 0.875, 0.880, 0.884, 0.889],
    'Recall': [0.865, 0.872, 0.877, 0.881, 0.886],
    'Macro-F1': [0.866, 0.873, 0.878, 0.882, 0.887],
    'Micro-F1': [0.869, 0.876, 0.881, 0.885, 0.890],
    'AUC-ROC': [0.878, 0.884, 0.889, 0.894, 0.899]
}

# 选择要展示的指标和对应的颜色
selected_metrics = ['Accuracy', 'Macro-F1', 'AUC-ROC']
colors = ['#dae6f1', '#78aac8', '#225b91']

# 创建图表
fig, ax = plt.subplots(figsize=(10, 6))

# 设置柱状图位置
x = np.arange(len(models))
width = 0.25  # 柱子的宽度

# 绘制柱状图
bars = []
for i, (metric, color) in enumerate(zip(selected_metrics, colors)):
    bar = ax.bar(x + (i-1)*width, metrics[metric], width,
                 label=metric, color=color)
    bars.append(bar)

# 设置图表样式
ax.set_ylabel('Performance Score', fontsize=10)
ax.set_ylim(0.85, 0.91)
ax.grid(True, linestyle='--', alpha=0.3, axis='y')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置x轴
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)

# 添加图例
ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(0.99, 0.99))

# 添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.001,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

for bar in bars:
    add_value_labels(bar)

# 添加标题
plt.title('Ablation Study of Different Components', fontsize=12, pad=15)

# 调整布局
plt.tight_layout()

# 保存为SVG
plt.savefig('ablation_study.svg', format='svg', dpi=300, bbox_inches='tight')
plt.close()

# 打印完整的性能指标表格
print("\nComplete Performance Metrics:")
for model in models:
    print(f"\n{model.replace('n', ' ')}")
    for metric, values in metrics.items():
        print(f"{metric}: {values[models.index(model)]:.3f}")
