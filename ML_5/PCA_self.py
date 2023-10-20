#データの読み込み
import pandas as pd 
from sklearn.datasets import load_iris 
dataset = load_iris() 
df_iris =  pd.DataFrame(dataset.data,columns=dataset.feature_names) 
df_iris['target'] = dataset.target  
y = dataset.target

#次元削減
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df_iris)
print(df_iris)

# 各主成分によってどの程度カバー出来ているかの割合(第一主成分，第二主成分)
print(pca.explained_variance_ratio_)

# 次元削減をdf_irisに適用する．
feature = pca.fit_transform(df_iris.iloc[:, :-1])    #ndarray  最後の列（target）を除いて次元削減
print(feature)
import numpy as np
feature_with_target = np.hstack((feature, np.array([y]).T))
import matplotlib.pyplot as plt 

fwt = feature_with_target
# カラーマップの色を定義
from matplotlib.colors import ListedColormap
colors = [  (1, 0, 0),   # 赤
            (0, 1, 0),   # 緑
            (0, 0, 1)]   # 青

# カラーマップを作成
cmap = ListedColormap(colors)

plt.scatter(fwt[:, 0], fwt[:,1], c=y , cmap=cmap, alpha=0.8, edgecolor='black')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of IRIS dataset')
#カスタム凡例テキストの使用
import matplotlib.patches as mpatches   #カスタム凡例テキストを使用するためにpatchesを使う
patch1 = mpatches.Patch(color='r', label='setosa')
patch2 = mpatches.Patch(color='g', label='versicolor')
patch3 = mpatches.Patch(color='b', label='virginica')
plt.legend(handles=[patch1,patch2,patch3],loc='upper right')
plt.savefig('PCA.pdf', format='pdf')       #出力したグラフをiris.pdfという名前で保存する
plt.show()