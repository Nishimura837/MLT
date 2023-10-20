import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Irisデータセットを読み込む
iris = datasets.load_iris()
X = iris.data

# データを["petal length", "petal width"]の2つの特徴量に減らす
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# K-meansクラスタリングを実行
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_reduced)
y_kmeans = kmeans.predict(X_reduced)


from matplotlib.colors import ListedColormap

colors = ["red", "green", "blue"]
cmap = ListedColormap(colors, name="custom")

# クラスタリング結果の散布図
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_kmeans, cmap=cmap)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('K-means Clustering Results')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

# 元のデータの散布図
plt.scatter(X[:, 2], X[:, 3], c=iris.target, cmap=cmap)
plt.title('Original Data')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()
#このコードは、Irisデータセットを2つの主成分にPCAで削減し、それからK-meansクラスタリングを実行します。最後に、クラスタリング結果と元のデータの2つの散布図を表示します。





