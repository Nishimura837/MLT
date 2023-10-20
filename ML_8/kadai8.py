import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


#Irisデータをsklearnから読み込み
iris = load_iris()


#辞書型に近い型でデータが入っているので必要箇所を指定してdataとcolumnsとしてdfで読み込み
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df1 = df.drop(columns = ['sepal length (cm)', 'sepal width (cm)']) #dropで指定したやつを削除
feature = iris.data
target = iris.target

#iris datasetの特徴量をpetal length, petal width の2つに減らす
X = df1.iloc[:, :2]

# K-meansクラスタリングを実行
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

X['y_kmeans'] = y_kmeans

#figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
fig = plt.figure()

#add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# クラスタリング結果の散布図
def set_marker(i):
    if i == 0:
        return "o"  # 丸
    elif i == 1:
        return "^"  # 三角
    else:
        return "s"  # 四角

for index, row in X.iterrows():
    marker = set_marker(row['y_kmeans'])
    ax1.scatter(row['petal length (cm)'], row['petal width (cm)'], c='grey', marker=marker)

ax1.set_title('K-means Clustering Results')
ax1.set_xlabel('Petal Length')
ax1.set_ylabel('Petal Width')


def set_color(i):
    if i == 0:
        return "r"  # blue
    elif i == 1:
        return "g"  # green
    else:
        return "b"  # red
    
color_list = list(map(set_color, iris.target))

#元データの散布図
ax2.scatter(X.iloc[:, 0], X.iloc[:, 1], c = color_list)
ax2.set_title('Original Data')
ax2.set_xlabel('Petal Length')
ax2.set_ylabel('Petal Width')

import matplotlib.patches as mpatches

# カスタム凡例テキストを使用するために、patchesを使う
patch1 = mpatches.Patch(color='r', label='setosa')
patch2 = mpatches.Patch(color='g', label='versicolor')
patch3 = mpatches.Patch(color='b', label='virginica')
fig.legend(handles=[patch1, patch2, patch3], loc='lower right')

fig.tight_layout() #グラフとグラフの間隔を開ける
plt.show()