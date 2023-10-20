import pandas as pd

from sklearn.datasets import load_iris 

#Irisデータをsklearnから読み込み
iris = load_iris()

#辞書型に近い型でデータが入っているので必要箇所を指定してdataとcolumnsとしてdfで読み込み
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

#最大値、最小値、平均値、中央値、標準偏差の出力
'''
print(df.max())
print(df.min())
print(df.mean()) 
print(df.quantile(q = [0.5])) #0.25で第一四分位数、0.75で第三四分位数
print(df.std())
'''
#代表値一覧での出力
'''
print(df.describe())

'''
df1 = pd.DataFrame(
    data={'最大値': [df['sepal length (cm)'].max(),df['sepal width (cm)'].max(),\
                     df['petal length (cm)'].max(),df['petal width (cm)'].max()],
          '最小値': [df['sepal length (cm)'].min(),df['sepal width (cm)'].min(),\
                     df['petal length (cm)'].min(),df['petal width (cm)'].min()],
          '平均値': [df['sepal length (cm)'].mean(),df['sepal width (cm)'].mean(),df['petal length (cm)'].mean(),df['petal width (cm)'].mean()],
          '中央値': [df['sepal length (cm)'].median(),df['sepal width (cm)'].median(),\
                     df['petal length (cm)'].median(),df['petal width (cm)'].median()],
          '標準偏差': [df['sepal length (cm)'].std(ddof=0),df['sepal width (cm)'].std(ddof=0),\
                       df['petal length (cm)'].std(ddof=0),df['petal width (cm)'].std(ddof=0)]},
                       
    index=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

print(df1)
df1.to_csv('basic_statistics.csv',encoding= 'utf_8_sig')

df2 = df.cov(ddof=0)

print(df2)
df2.to_csv('covariance.csv', encoding= 'utf_8_sig')

df3 = df.corr()

print(df3)
df3.to_csv('correlationcoefficient.csv', encoding= 'utf_8_sig')