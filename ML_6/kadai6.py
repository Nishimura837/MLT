import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

#Irisデータをsklearnから読み込み
iris = load_iris()


#辞書型に近い型でデータが入っているので必要箇所を指定してdataとcolumnsとしてdfで読み込み
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

feature = iris.data
target = iris.target

#標準化のための読み込み
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iris_scaled_df = scaler.fit_transform(iris_df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state = 0)
iris_std_df = pca.fit_transform(iris_scaled_df)

#カラーマップの設定
from matplotlib.colors import ListedColormap
cmap = ListedColormap(('red', 'green', 'blue'))

plt.figure(figsize=(8, 8))
plt.scatter(iris_std_df[:, 0], iris_std_df[:, 1], alpha = 0.4, c=target, cmap=cmap)

# カスタム凡例テキストを使用するために、patchesを使う
import matplotlib.patches as mpatches
patch1 = mpatches.Patch(color='blue', label='setosa')
patch2 = mpatches.Patch(color='green', label='versicolor')
patch3 = mpatches.Patch(color='red', label='virginica')
plt.legend(handles=[patch1, patch2, patch3], loc='lower right')
plt.title('iris std')
plt.show()


import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
X = iris_std_df
y = target

#減らしたものをトレーニングデータ：テストデータ＝7：3にする

                                                                    #test_sizeで分割の割合指定0~1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#モデルの作成
tree = DecisionTreeClassifier(max_depth = 3) #最大深さ
tree.fit(X_train, y_train)

#モデルの可視化
fig, ax = plt.subplots(figsize=(10, 10))
plot_tree(tree, feature_names=iris.feature_names, filled=True)
plt.show()

#性能評価
print('正答率: {:.3f}'.format(tree.score(X_test, y_test)))


#決定境界の範囲設定
X_min, X_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
X_mesh, y_mesh = np.meshgrid(np.arange(X_min, X_max, 0.01),
                                   np.arange(y_min, y_max, 0.01))

Z = tree.predict(np.c_[X_mesh.ravel(), y_mesh.ravel()])
Z = Z.reshape(X_mesh.shape)

#決定境界の描画 (contourfのfを消すと色が消える)
plt.contourf(X_mesh, y_mesh, Z, alpha=0.4, cmap=cmap)


#データの可視化
plt.scatter(X[1:50, 0], X[1:50, 1], c='r',marker='s', edgecolor='k', label = 'Setosa')
plt.scatter(X[51:100, 0], X[51:100, 1], c='g', marker='o',edgecolor='k', label = 'Versicolor')
plt.scatter(X[101:150, 0], X[101:150, 1], c='b', marker='x',edgecolor='k', label = 'Varginica')
plt.title("iris std clasify")
plt.legend(loc = 'best')
plt.show()
