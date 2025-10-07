import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = {
    'activation_quant_bits': [1, 2, 4, 8, 16],
    'weight_quant_bits': [1, 2, 4, 8, 16],
    'compression_ratio': [26.10, 14.42, 7.61, 3.91, 1.99],
    'model_size_mb': [0.34, 0.62, 1.18, 2.29, 8.97],
    'quantized_acc': [1.0, 1.0, 18.63, 64.29, 64.25],
    'label': ['A', 'B', 'C', 'D', 'E']
}

df = pd.DataFrame(data)
features = ['activation_quant_bits', 'weight_quant_bits', 'compression_ratio', 'model_size_mb', 'quantized_acc']

# Normalize data for plotting
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[features] = scaler.fit_transform(df[features])

# Plot
plt.figure(figsize=(12, 6))
parallel_coordinates(df_scaled, 'label', colormap=plt.get_cmap("Set2"))

ax = plt.gca()

num_vars = len(features)
xticks = np.arange(num_vars)

# Hide all yticks on the plot (default 0-1 ticks)
ax.set_yticks([])

# Add original scale ticks manually per axis
for i, feature in enumerate(features):
    min_val = df[feature].min()
    max_val = df[feature].max()

    ticks = [0, 0.25, 0.5, 0.75, 1]
    tick_labels = [f"{min_val + t*(max_val - min_val):.1f}" for t in ticks]

    x = xticks[i]

    for t, tl in zip(ticks, tick_labels):
        ax.text(x, t, tl, ha='right', va='center', fontsize=8)

plt.xticks(xticks, features, rotation=20, ha='right')
plt.title("Parallel Coordinates Plot with Original Scale Ticks Only")
plt.grid(True)
plt.tight_layout()
plt.savefig("wandb_plot.png", bbox_inches='tight')
plt.close()
