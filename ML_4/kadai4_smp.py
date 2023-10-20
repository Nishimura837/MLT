# 必要なライブラリをインポート
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Irisデータセットを読み込む
iris = load_iris()
X = iris.data[:, [2, 3]]  # 特徴量をpetal lengthとpetal widthに減らす
y = iris.target

# データセットをトレーニングデータとテストデータに分割 (7:3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 決定木モデルを作成
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# テストデータを用いて予測を行い、正解率を計算
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"分類の正解率: {accuracy:.2f}")

# 決定境界をプロット
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='k')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Irisデータセットのクラス分類")
plt.show()






