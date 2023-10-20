import pandas as pd

from sklearn.datasets import load_iris 

#Irisデータをsklearnから読み込み
iris = load_iris()


#辞書型に近い型でデータが入っているので必要箇所を指定してdataとcolumnsとしてdfで読み込み
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['target'] = iris.target
df['No'] = range(1, len(df.index) + 1)

print(df)
print(df.columns) #行数の出力
print(df.index)  #列数の出力
#散布図の作成
import matplotlib as mpl
import matplotlib.pyplot as plt

