import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Irisデータセットを読み込む
iris = load_iris()
X = iris.data
y = iris.target

# PCAを適用して2次元に削減
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# クラスごとに色を設定
colors = ['red', 'green', 'blue']
lw = 2

# グラフを描画
plt.figure(figsize=(8, 6))
for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()