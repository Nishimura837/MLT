import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Iris データセットを読み込む
iris = load_iris()
X = iris.data[:, [2, 3]]  # petal length, petal width
y = iris.target

# DataFrame を作成
df = pd.DataFrame(data=X, columns=['petal length', 'petal width'])
df['target'] = y

# 決定木モデルの訓練
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(df[['petal length', 'petal width']], df['target'])

#作成したモデル木の可視化
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
plot_tree(clf, feature_names=y, filled=True)
plt.show()

# 決定境界を描画するためのグリッドを作成
x_min, x_max = df['petal length'].min() - 1, df['petal length'].max() + 1
y_min, y_max = df['petal width'].min() - 1, df['petal width'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
print(type(X))
print(X)
print(y)

# 各グリッドポイントに対して予測を行い、決定境界を計算
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

d = df.iloc
# データポイントと決定境界をプロット
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(d[0:50, 0], d[:50, 1], color='r', marker='o', label='setosa')
plt.scatter(d[50:100, 0], d[50:100, 1], color='y', marker='+', label='versicolor')
plt.scatter(d[100:, 0], d[100:, 1], color='b', marker='x', label='virginica')
plt.xlabel("petal length(cm)")
plt.ylabel("petal width(cm)")
plt.legend(loc='upper right')
plt.show()
