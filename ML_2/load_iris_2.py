#これは回答ではない

from sklearn.datasets import load_iris 
import pandas as pd 
dataset = load_iris()
df_iris =  pd.DataFrame(dataset.data,columns=dataset.feature_names) 
df_iris['target'] = dataset.target  
df_iris['No'] = range(1,len(df_iris.index)+1)   #1から始まる連番を設定した列を追加する
print(df_iris)


print(dir(dataset))     #dir関数により属性を取得する
print(list(dataset.target_names))   #datasetに格納されているアヤメの種類を確認する
print(dataset.feature_names)    #各列に何が格納されているかを確認する


#x軸にはがく片の長さ、y軸にはがく片の幅が来る散布図の作成
import matplotlib.pyplot as plt

x = dataset.data
y = dataset.target

plt.scatter(x[:50, 0], x[:50, 1], color='r', marker='o', label='setosa')
plt.scatter(x[50:100, 0], x[50:100, 1], color='g', marker='+', label='versicolor')
plt.scatter(x[100:, 0], x[100:, 1], color='b', marker='x', label='virginica')
plt.title("Iris Plants Database")
plt.xlabel("sepal length(cm)")
plt.ylabel("sepal width(cm)")
plt.legend()
plt.show()
