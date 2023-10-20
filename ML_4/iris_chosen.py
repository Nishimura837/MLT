##決定木の作成、可視化、予測の実行、正解率の表示まで

#データの読み込み
import pandas as pd 
from sklearn.datasets import load_iris 
dataset = load_iris() 
df_iris =  pd.DataFrame(dataset.data,columns=dataset.feature_names) 
df_iris['target'] = dataset.target 
print(df_iris) 
print (df_iris.columns) #列名を取得
#列名を指定して削除
df1=df_iris.drop(columns=['sepal length (cm)', 'sepal width (cm)'])
#削除後のデータフレームの表示
print(df1)

#データを図示する
d= df1.iloc
import matplotlib.pyplot as plt 
plt.scatter(d[0:50, 0], d[:50, 1], color='r', marker='o', label='setosa')
plt.scatter(d[50:100, 0], d[50:100, 1], color='g', marker='+', label='versicolor')
plt.scatter(d[100:, 0], d[100:, 1], color='b', marker='x', label='virginica')
plt.xlabel("petal length(cm)")
plt.ylabel("petal width(cm)")
plt.legend(loc='lower right')
plt.show()

#トレーニングデータとテストデータに分類する
from sklearn.model_selection import train_test_split
df1_train, df1_test = train_test_split(df1,test_size=0.3,random_state=42) \
    #train_test_split(namelist, test_size=0.3)ここでは30%がテストデータ,random_stateはシード値
print("train_data") 
print(df1_train)
print("test_data")
print(df1_test)

print("df1_0")
print(df1.iloc[:,0])
print("df1_1")
print(df1.iloc[:,1])
print("df1_2")
print(df1.iloc[:,2])

#モデル（決定木）を作成
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(df1_train.iloc[:,0:1], df1_train.iloc[:,2])

#作成したモデル木の可視化
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree
plt.figure(figsize=(15,10))
plot_tree(clf, feature_names=df1_train.target, filled=True)
plt.show()

#作成したモデル（決定木）をもちいて予測を実行
predicted = clf.predict(df1_test.iloc[:,0:1])
print(predicted)

#識別率の確認
print(sum(predicted == df1_test.target)/len(df1_test.target))








