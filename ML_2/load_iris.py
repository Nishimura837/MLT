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

fig,axes = plt.subplots(2,2)

def set_color(i):
    if i == 0:
        return "b"  # blue
    elif i == 1:
        return "g"  # green
    else:
        return "r"  # red
    
color_list = list(map(set_color, iris.target))

#項目ごとの散布図
df.plot.scatter(x='No', y='sepal length (cm)', alpha=1.0, c=color_list, ax=axes[0,0], title='sepal length')

df.plot.scatter(x='No', y='sepal width (cm)', alpha=1.0, c=color_list, ax=axes[0,1], title='sepal width')

df.plot.scatter(x='No', y='petal length (cm)', alpha=1.0, c=color_list, ax=axes[1,0],title='petal length')

df.plot.scatter(x='No', y='petal width (cm)', alpha=1.0, c=color_list, ax=axes[1,1],title='petal width')

import matplotlib.patches as mpatches
 
# カスタム凡例テキストを使用するために、patchesを使う
patch1 = mpatches.Patch(color='blue', label='setosa')
patch2 = mpatches.Patch(color='green', label='versicolor')
patch3 = mpatches.Patch(color='red', label='virginica')
plt.legend(handles=[patch1, patch2, patch3], loc='lower right')

'''
#targetごとの散布図
import seaborn as sns
# JupyterLab で実行する際は、この行を書くことで描画できるようになる。

sns.pairplot(df, hue="target")
'''
plt.tight_layout() #グラフとグラフの間隔を開ける
plt.show() #グラフの出力
