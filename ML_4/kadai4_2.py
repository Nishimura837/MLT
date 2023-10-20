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

#減らしたものをトレーニングデータ：テストデータ＝7：3にする

                                                                    #test_sizeで分割の割合指定0~1
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=42)

#モデルの作成
tree = DecisionTreeClassifier(max_depth = 3) #最大深さが3
tree.fit(X_train, y_train)

#モデルの可視化
fig, ax = plt.subplots(figsize=(10, 10))
plot_tree(tree, feature_names=iris.feature_names, filled=True)

#予測値の出力
y_pred = tree.predict(X_test)

print(y_pred[:10])
print(y_test[:10])

#性能評価
print('正答率: {:.3f}'.format(tree.score(X_test, y_test)))

#カラーマップの設定
from matplotlib.colors import ListedColormap
cmap = ListedColormap(('red', 'blue', 'green'))

#決定境界の範囲設定
X_min, X_max = X[:, 2].min()-1, X[:, 3].max()+1
y_min, y_max = y[:, 2].min()-1, y[:, 3].max()+1
X_mesh, y_mesh = np.meshgrid(np.arange(X_min, X_max, 0.01),
                                   np.arange(y_min, y_max, 0.01))

Z = tree.predict(np.array([X_mesh.ravel(), y_mesh.ravel()]).T)
Z = Z.reshape(X_mesh.shape)

#決定境界の描画 (contourfのfを消すと色が消える)
plt.contourf(X_mesh, y_mesh, Z, alpha=0.4, cmap=cmap)
plt.xlim(X_mesh.min(), X_mesh.max())
plt.ylim(y_mesh.min(), y_mesh.max())


#データの可視化
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(X_train[y_train == 0, 2], X_train[y_train == 0, 3], 
           marker = 'o', c='r' ,label = 'Setosa')

ax.scatter(X_train[y_train == 1, 2], X_train[y_train == 1, 3],
           marker = 'x', c='g' ,label = 'Versicolor')

ax.scatter(X_train[y_train == 2, 2], X_train[y_train == 2, 3],
           marker = 's', c='b' ,label = 'Varginica')

ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal width')
ax.legend(loc = 'best')
plt.show()
