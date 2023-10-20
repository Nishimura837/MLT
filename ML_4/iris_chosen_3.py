##回答

#データの読み込み
import pandas as pd 
from sklearn.datasets import load_iris 
dataset = load_iris() 
df_iris =  pd.DataFrame(dataset.data,columns=dataset.feature_names) 
df_iris['target'] = dataset.target 
#print(df_iris) 
#print (df_iris.columns) #列名を取得
#列名を指定して削除
df1=df_iris.drop(columns=['sepal length (cm)', 'sepal width (cm)'])
#削除後のデータフレームの表示
#print(df1)

#トレーニングデータとテストデータに分類する
from sklearn.model_selection import train_test_split
df1_train, df1_test = train_test_split(df1,test_size=0.3,random_state=42) 
    #train_test_split(namelist, test_size=0.3)ここでは30%がテストデータ,random_stateはシード値


#モデル（決定木）を作成
from sklearn import tree
#モデルを定義
clf = tree.DecisionTreeClassifier(max_depth=3)
#モデルの学習
clf.fit(df1_train.iloc[:,0:2], df1_train.iloc[:,2])

#作成したモデル木の可視化
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
plot_tree(clf, feature_names=df1_train.target, filled=True)
plt.show()

#作成したモデル（決定木）をもちいて予測を実行
predicted = clf.predict(df1_test.iloc[:,0:2])
print(predicted)

#識別率の確認
print("正解率 =")
print(sum(predicted == df1_test.target)/len(df1_test.target))

#これまで使ってきたデータフレームをnumpyに変換する
import numpy as np 
f_df1 = df1.values
#print(f_df1.dtype)
#整数型に変換
i_df1 = f_df1.astype(int)
#print(i_df1.dtype)

#決定境界の可視化
X = f_df1[:,0:2]        #petal length , petal width 
y = i_df1[:,2]         #target


#決定境界を描画するためのグリッドを作成
x_min, x_max = df1.iloc[:,0].min() - 1, df1.iloc[:,0].max() + 1
y_min, y_max = df1.iloc[:,1].min() - 1, df1.iloc[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))


# 各グリッドポイントに対して予測を行い、決定境界を計算
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#カラーマップの作成
from matplotlib.colors import ListedColormap

# カラーマップの色を定義
colors = [  (1, 0, 0),   # 赤
            (0, 1, 0),   # 緑
            (0, 0, 1)]   # 青

# カラーマップを作成
cmap = ListedColormap(colors)

#データを図示する
d= df1.iloc
import matplotlib.pyplot as plt 
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
plt.scatter(d[0:50, 0], d[:50, 1], color='r', marker='s', edgecolor='black', label='setosa')
plt.scatter(d[50:100, 0], d[50:100, 1], color='g', marker='x', edgecolor='black', label='versicolor')
plt.scatter(d[100:, 0], d[100:, 1], color='b', marker='o', edgecolor='black', label='virginica')
plt.xlabel("petal length(cm)")
plt.ylabel("petal width(cm)")
plt.legend(loc='upper right')
plt.show()