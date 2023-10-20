import matplotlib.pyplot as plt
import numpy as np

# マーカーを指定するための関数を定義
def set_marker(value):
    if value == 0:
        return "s"  # 正方形のマーカー
    elif value == 1:
        return "^"  # 上向き三角形のマーカー
    else:
        return "o"  # その他のマーカー

# サンプルデータ生成
np.random.seed(42)
x = np.random.rand(10)
y = np.random.rand(10)
z = [0, 1, 2, 1, 0, 1, 0, 2, 2, 0]

# 散布図をプロット
for xi, yi, zi in zip(x, y, z):
    marker = set_marker(zi)
    plt.scatter(xi, yi, marker=marker)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot with Custom Markers")
plt.show()
