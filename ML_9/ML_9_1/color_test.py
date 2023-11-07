import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ダミーデータを作成する例
data = {
    '行名': ['A', 'B', 'C', 'A', 'D', 'E', 'B', 'C', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'],
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'Y': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5]
}

df = pd.DataFrame(data)

# 色分けする列名（ここでは'行名'列）を指定
plot_column = '行名'

# カラーマップを選択
cmap = plt.get_cmap('tab20')

# プロット
plt.figure(figsize=(8, 6))

unique_values = df[plot_column].unique()
colors = cmap(np.linspace(0, 1, len(unique_values)))

for value, color in zip(unique_values, colors):
    subset = df[df[plot_column] == value]
    plt.scatter(subset['X'], subset['Y'], label=value, color=color)

plt.legend()
plt.xlabel('X軸')
plt.ylabel('Y軸')
plt.title('色分けされたプロット')
plt.grid(True)

plt.show()