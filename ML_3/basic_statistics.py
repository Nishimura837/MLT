#データの読み込み
import pandas as pd 
from sklearn.datasets import load_iris 
dataset = load_iris() 
df_iris =  pd.DataFrame(dataset.data,columns=dataset.feature_names) 

'''
#describe関数によるデータ概要の取得
print(df_iris.describe())
'''

'''
#最大値、最小値、平均値、中央値、標準偏差の取得
print('最大値')
print(df_iris.max())
print('最小値')
print(df_iris.min())
print('平均値')
print(df_iris.mean())
print('中央値')
print(df_iris.median())
print('標準偏差')
print(df_iris.std(ddof=0))
print('共分散')
print(df_iris.cov(ddof=0))
'''
#最大値、最小値、平均値、中央値、標準偏差をデータに持つデータフレームdf1を作成
df1 = pd.DataFrame(
    data={  '最大値':[df_iris['sepal length (cm)'].max(),df_iris['sepal width (cm)'].max(),\
                df_iris['petal length (cm)'].max(),df_iris['petal width (cm)'].max()],
            '最小値':[df_iris['sepal length (cm)'].min(),df_iris['sepal width (cm)'].min(),\
                df_iris['petal length (cm)'].min(),df_iris['petal width (cm)'].min()],
            '平均値':[df_iris['sepal length (cm)'].mean(),df_iris['sepal width (cm)'].mean(),\
                df_iris['petal length (cm)'].mean(),df_iris['petal width (cm)'].mean()],
            '中央値':[df_iris['sepal length (cm)'].median(),df_iris['sepal width (cm)'].median(),\
                df_iris['petal length (cm)'].median(),df_iris['petal width (cm)'].median()],
            '標準偏差':[df_iris['sepal length (cm)'].std(ddof=0),df_iris['sepal width (cm)'].std(ddof=0),\
                df_iris['petal length (cm)'].std(ddof=0),df_iris['petal width (cm)'].std(ddof=0)]},
    index=[ 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
)
#df1をcsvファイルとして出力
print(df1)
df1.to_csv('basic_statistics.csv',encoding='utf_8_sig')

#共分散をデータに持つデータフレームdf2を作成
'''
df2 = pd.DataFrame(
    data={'共分散':[df_iris.cov(ddof=0)]}
)
'''
df2 = df_iris.cov(ddof=0)
#df2をcsvファイルとして出力
print(df2)
df2.to_csv('covariance.csv',encoding='utf_8_sig')

#相関係数をデータに持つデータフレームdf3を作成
'''
df3 =pd.DataFrame(
    data={'相関係数':[df_iris.corr()]}
)
'''
df3 = df_iris.corr()
#df3をcsvファイルとして出力
print(df3)
df3.to_csv('correlationcoefficient.csv',encoding='utf_8_sig')


'''
#csvファイルへの出力
with open('python_kadai/basic_statistic.csv','w') as f:
    print('最大値')
    print(df_iris.max())
    print('最小値')
    print(df_iris.min())
    print('平均値')
    print(df_iris.mean())
    print('中央値')
    print(df_iris.median())
    print('標準偏差')
    print(df_iris.std())
'''



