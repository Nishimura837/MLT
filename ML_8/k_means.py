import pandas as pd 
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 

dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target
print(df)

#列名を指定して削除
df1 = df.drop(columns=['sepal length (cm)', 'sepal width (cm)'])
print(df1)

#教師なし学習のk-meansを使ってクラスタリングを行う
X = df1.iloc[:,:2]
print(X)
Y = df1.iloc[:,2]

kmeans = KMeans(n_clusters=3, init='random', n_init='auto', random_state=42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)
#y_kmeansをデータフレーム化
z = pd.DataFrame(data=y_kmeans)

X['y_kmeans'] = z
print(X)

#クラスタリングしたデータをプロットする
# マーカーを指定するための関数を定義
def set_marker(value):
    if value == 0:
        return "o"  # 丸のマーカー
    elif value == 1:
        return "^"  # 上向き三角形のマーカー
    else:
        return "s"  # 正方形のマーカー

# 散布図をプロット
for index, row in X.iterrows():
    marker = set_marker(row['y_kmeans'])
    plt.scatter(row['petal length (cm)'], row['petal width (cm)'], \
        marker=marker, color='grey', edgecolor='black')
plt.grid(color='grey', linestyle=':', linewidth=0.3)
plt.title("clustering result")
plt.xlabel("petal length(cm)")
plt.ylabel("petal width(cm)")
plt.show()

#元データの図示
d= df1.iloc
plt.scatter(d[:50, 0], d[:50, 1], color='r', marker='s', edgecolor='black', label='setosa')
plt.scatter(d[50:100, 0], d[50:100, 1], color='g', marker='^', edgecolor='black', label='versicolor')
plt.scatter(d[100:, 0], d[100:, 1], color='b', marker='o', edgecolor='black', label='virginica')
plt.grid(color='grey', linestyle=':', linewidth=0.3)
plt.title("original data")
plt.xlabel("petal length(cm)")
plt.ylabel("petal width(cm)")
plt.legend(loc='lower right')
plt.show()

