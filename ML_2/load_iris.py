#データの読み込み
import pandas as pd 
from sklearn.datasets import load_iris 
dataset = load_iris() 
df_iris =  pd.DataFrame(dataset.data,columns=dataset.feature_names) 
df_iris['target'] = dataset.target  
df_iris['No'] = range(1,len(df_iris.index)+1)   #1から始まる連番を設定した列を追加する
list(dataset.target_names)
print(df_iris) 
print (df_iris.columns) #列名を取得
print(df_iris.index)    #行名を取得
#条件を指定して行、列を取得する
flag = df_iris["target"].isin(["0"]) 
print(df_iris[flag].head())
#散布図の作成
import matplotlib.pyplot as plt 
#項目ごとの散布図

def set_color(l):   #def　名前(引数) 　命令の関数
    if l == 0: 
        return "b"  # blue
    elif l == 1:
        return "g"  # green
    else:
        return "r"  # red
color_list = list(map(set_color, df_iris.target))   #map(関数名,処理対象)　map関数はobjectとしてしかないからそれをlist化する
_,axes = plt.subplots(2,2)                          #２行２列のグラフ領域を設定
df_iris.plot.scatter(x='No', y='sepal length (cm)',c=color_list,ax=axes[0,0], title='sepal length')
df_iris.plot.scatter(x='No', y='sepal width (cm)', c=color_list,ax=axes[0,1], title='sepal width')
df_iris.plot.scatter(x='No', y='petal length (cm)', c=color_list,ax=axes[1,0], title='petal length')
df_iris.plot.scatter(x='No', y='petal width (cm)', c=color_list,ax=axes[1,1], title='petal width') 
#カスタム凡例テキストの使用
import matplotlib.patches as mpatches   #カスタム凡例テキストを使用するためにpatchesを使う
patch1 = mpatches.Patch(color='b', label='setosa')
patch2 = mpatches.Patch(color='g', label='versicolor')
patch3 = mpatches.Patch(color='r', label='virginica')
plt.legend(handles=[patch1,patch2,patch3],loc='lower right')
plt.tight_layout()  #タイトルが被らないようにする
plt.savefig('iris.pdf', format='pdf')       #出力したグラフをiris.pdfという名前で保存する
plt.show()

"""
#保存先を指定してファイルを作成
import os 
dirname  = 'dir001/'    #dirnameをresult という名前にする
os.makedirs(dirname, exist_ok=True)    #makedirs()でディレクトリを新規作成する
filename = dirname + 'img.png'
plt.savefig(filename)

plt.show()
"""

#ーーーーー様々なグラフの書き方ーーーーー


# #targetごとの散布図
# import seaborn as sns
# sns.pairplot(df_iris, hue="target")
# plt.show()

# #折れ線グラフの描画
# import matplotlib.pyplot as plt 
# df_iris.plot(figsize=(10,5))
# plt.show() 

# df_iris.plot(legend=False,subplots=True)
# plt.show() 

# #ヒストグラムの描画
# df_iris.hist(figsize=(10,10))
# plt.tight_layout() 
# plt.show() 

# #棒グラフの描画
# mean = df_iris.groupby('target')['sepal length(cm)'].mean() 
# mean.plot.bar() 
# plt.show() 
