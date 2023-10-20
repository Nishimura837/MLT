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


kmeans = KMeans(n_clusters=3, init='random', n_init='auto', random_state=42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)
#y_kmeansをデータフレーム化
z = pd.DataFrame(data=y_kmeans)

X['y_kmeans'] = z
print(X)


fig = plt.figure(figsize=(10, 4))

# # フィギュア内のサブプロットの間隔を縦横0.5の分量で空ける
# plt.subplots_adjust(wspace=0.5, hspace=0.5)


#クラスタリングしたデータをプロットする
# マーカーを指定するための関数を定義
def set_marker(value):
    if value == 0:
        return "o"  # 丸のマーカー
    elif value == 1:
        return "^"  # 上向き三角形のマーカー
    else:
        return "s"  # 正方形のマーカー

fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1行2列のサブプロット
plt.subplots_adjust(wspace=0.4)

# 散布図をプロット
for index, row in X.iterrows():
    marker = set_marker(row['y_kmeans'])
    axes[0].scatter(row['petal length (cm)'], row['petal width (cm)'], \
        marker=marker, color='grey', edgecolor='black')
    
axes[0].grid(color='grey', linestyle=':', linewidth=0.3)
axes[0].set_title("clustering result")
axes[0].set_xlabel("petal length(cm)")
axes[0].set_ylabel("petal width(cm)")


#元データの図示
d= df1.iloc
axes[1].scatter(d[0:50, 0], d[:50, 1], color='r', marker='s', edgecolor='black', label='setosa')
axes[1].scatter(d[50:100, 0], d[50:100, 1], color='g', marker='^', edgecolor='black', label='versicolor')
axes[1].scatter(d[100:, 0], d[100:, 1], color='b', marker='o', edgecolor='black', label='virginica')
axes[1].grid(color='grey', linestyle=':', linewidth=0.3)
axes[1].set_title("original data")
axes[1].set_xlabel("petal length(cm)")
axes[1].set_ylabel("petal width(cm)")
axes[1].legend(loc='lower right')

plt.tight_layout()
# plt.show()
plt.savefig("/home/gakubu/デスクトップ/python_git/my_project/ML_8/k_means.pdf", format='pdf')       #出力したグラフをk_means.pdfという名前で保存する
