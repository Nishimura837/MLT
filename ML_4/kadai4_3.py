import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


#Irisデータをsklearnから読み込み
iris = load_iris()


#辞書型に近い型でデータが入っているので必要箇所を指定してdataとcolumnsとしてdfで読み込み
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

feature = iris.data
target = iris.target

#iris datasetの特徴量をpetal length, petal width の2つに減らす
X = feature[:, [2,3]]
y = target
"""print(X)"""
#減らしたものをトレーニングデータ：テストデータ＝7：3にする

                                                                    #test_sizeで分割の割合指定0~1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#モデルの作成
tree = DecisionTreeClassifier(max_depth = 3) #最大深さが3
tree.fit(X_train, y_train)

#モデルの可視化
fig, ax = plt.subplots(figsize=(10, 10))
plot_tree(tree, feature_names=iris.feature_names, filled=True)
plt.show()

#性能評価
print('正答率: {:.3f}'.format(tree.score(X_test, y_test)))

#カラーマップの設定
from matplotlib.colors import ListedColormap
cmap = ListedColormap(('red', 'green', 'blue'))

#決定境界の範囲設定
X_min, X_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
X_mesh, y_mesh = np.meshgrid(np.arange(X_min, X_max, 0.01),
                                   np.arange(y_min, y_max, 0.01))

Z = tree.predict(np.c_[X_mesh.ravel(), y_mesh.ravel()])
Z = Z.reshape(X_mesh.shape)

#決定境界の描画 (contourfのfを消すと色が消える)
plt.contourf(X_mesh, y_mesh, Z, alpha=0.4, cmap=cmap)
#plt.xlim(X_mesh.min(), X_mesh.max())
#plt.ylim(y_mesh.min(), y_mesh.max())

#データの可視化
plt.scatter(X[1:50, 0], X[1:50, 1], c='r',marker='s', edgecolor='k', label = 'Setosa')
plt.scatter(X[51:100, 0], X[51:100, 1], c='g', marker='o',edgecolor='k', label = 'Versicolor')
plt.scatter(X[101:150, 0], X[101:150, 1], c='b', marker='x',edgecolor='k', label = 'Varginica')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris dataset clasify")
plt.legend(loc = 'best')
plt.show()
