import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib as mpl

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'Arial'

# Define colors from your palette
colors = ['#dae6f1', '#b3cede', '#78aac8', '#4884af', '#225b91', '#225b91']

# Create figure and subplots
fig = plt.figure(figsize=(14, 10))
fig.suptitle('DermaTrans Hyperparameter Studies', fontsize=24, fontweight='bold', y=0.95)

# 1. Learning Rate (Line Chart) - Top Left
ax1 = plt.subplot2grid((2, 2), (0, 0))
lr_values = ['1e-5', '1e-4', '1e-3']
accuracy = [0.880, 0.892, 0.871]

ax1.plot(lr_values, accuracy, marker='o', markersize=10, linewidth=2, color=colors[3])
for i, (x, y) in enumerate(zip(lr_values, accuracy)):
    ax1.scatter(x, y, s=120, color=colors[0], edgecolor=colors[4], linewidth=2, zorder=5)
    ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                 xytext=(0, -15), ha='center', fontsize=11)

ax1.set_ylim(0.86, 0.90)
ax1.set_title('Learning Rate', fontsize=18, fontweight='bold', pad=15)
ax1.set_xlabel('Learning Rate', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)

# 2. Batch Size (Radar Chart) - Top Right
ax2 = plt.subplot2grid((2, 2), (0, 1), polar=True)

# Data for the radar chart
batch_sizes = ['16', '32', '64', '128', '256']
batch_accuracy = [0.885, 0.888, 0.892, 0.884, 0.879]
# Convert to range 0-1 for better visualization (normalizing values)
normalized_accuracy = [(x - 0.87) / 0.03 for x in batch_accuracy]  # Normalize between 0.87 and 0.90

# Number of variables
N = len(batch_sizes)
# Angle of each axis
angles = [n / N * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop
normalized_accuracy += normalized_accuracy[:1]  # Close the loop

# Draw polygon
ax2.plot(angles, normalized_accuracy, linewidth=2, linestyle='solid', color=colors[3])
ax2.fill(angles, normalized_accuracy, alpha=0.4, color=colors[2])

# Draw axis lines
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(batch_sizes, fontsize=12)

# Remove radial labels and add custom ones
ax2.set_yticklabels([])
for i, acc in enumerate(batch_accuracy[:-1]):
    angle_rad = angles[i]
    if 0 <= angle_rad <= np.pi / 2 or 3 * np.pi / 2 <= angle_rad <= 2 * np.pi:
        ha = 'left'
    else:
        ha = 'right'

    if np.pi / 4 <= angle_rad <= 3 * np.pi / 4 or 5 * np.pi / 4 <= angle_rad <= 7 * np.pi / 4:
        va = 'top'
    else:
        va = 'bottom'

    offset = 0.2  # Adjust this value as needed
    x = (normalized_accuracy[i] + offset) * np.cos(angle_rad)
    y = (normalized_accuracy[i] + offset) * np.sin(angle_rad)
    # Fix: Remove the second xy parameter
    ax2.annotate(f"{batch_accuracy[i]:.3f}", xy=(x, y), fontsize=10, ha=ha, va=va)

ax2.set_title('Batch Size', fontsize=18, fontweight='bold', y=1.1)

# Add a note for the best batch size
ax2.annotate('Best: 64 (0.892)', xy=(0.5, -0.15), xycoords='axes fraction',
             fontsize=14, ha='center', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# 3. Optimizer (Bar Chart) - Bottom Left
ax3 = plt.subplot2grid((2, 2), (1, 0))
optimizers = ['SGD', 'Adam', 'AdamW', 'RMSprop']
opt_accuracy = [0.865, 0.892, 0.881, 0.875]

bars = ax3.bar(optimizers, opt_accuracy, width=0.6, edgecolor='black', linewidth=1)

# Color each bar differently
for i, bar in enumerate(bars):
    bar.set_color(colors[i % len(colors)])

for i, v in enumerate(opt_accuracy):
    ax3.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=11)

ax3.set_ylim(0.85, 0.90)
ax3.set_title('Optimizer', fontsize=18, fontweight='bold', pad=15)
ax3.set_xlabel('Optimizer Type', fontsize=14)
ax3.set_ylabel('Accuracy', fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7, axis='y')

# 4. Dropout Rate (Area Chart) - Bottom Right
ax4 = plt.subplot2grid((2, 2), (1, 1))
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
dropout_accuracy = [0.889, 0.892, 0.887, 0.883, 0.879]

# Create area chart
ax4.fill_between(dropout_rates, dropout_accuracy, 0.87, alpha=0.4, color=colors[2], edgecolor=colors[4], linewidth=2)
ax4.plot(dropout_rates, dropout_accuracy, 'o-', color=colors[4], markersize=8, linewidth=2)

for i, (x, y) in enumerate(zip(dropout_rates, dropout_accuracy)):
    ax4.annotate(f"{y:.3f}", xy=(x, y), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=11)

ax4.set_xlim(0.05, 0.55)
ax4.set_ylim(0.875, 0.895)
ax4.set_title('Dropout Rate', fontsize=18, fontweight='bold', pad=15)
ax4.set_xlabel('Dropout Rate', fontsize=14)
ax4.set_ylabel('Accuracy', fontsize=14)
ax4.grid(True, linestyle='--', alpha=0.7)

# Add a highlighted region for the best dropout
ax4.axvspan(0.15, 0.25, alpha=0.2, color=colors[5])
ax4.annotate('Optimal\nRange', xy=(0.2, 0.877), xytext=(0.2, 0.876),
             ha='center', fontsize=12, color=colors[5],
             arrowprops=dict(facecolor=colors[5], shrink=0.05, width=2, headwidth=8))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save as SVG
plt.savefig('hyperparameter_visualization.svg', format='svg', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()