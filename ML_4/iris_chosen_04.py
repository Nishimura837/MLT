import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mlxtend.plotting import plot_decision_regions

# Irisデータセットのロード
iris = load_iris()
X = iris.data[:, [2, 3]]  # petal lengthとpetal widthの2つの特徴量を選択
y = iris.target

# データセットをトレーニングデータとテストデータに分割（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 決定木アルゴリズムを用いたクラス分類
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X_train, y_train)

# 決定木の可視化
plt.figure(figsize=(10, 7))
plot_tree(clf, filled=True, feature_names=["Petal Length", "Petal Width"], class_names=iris.target_names)
plt.show()

# 予測の実行
y_pred = clf.predict(X_test)
print(y_pred)
# 分類の正解率を表示
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# 決定境界とデータセットを可視化
plot_decision_regions(X, y, clf=clf, legend=2)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Decision Boundary and Data Points')
plt.show()