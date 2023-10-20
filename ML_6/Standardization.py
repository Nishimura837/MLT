import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches   #カスタム凡例テキストを使用するためにpatchesを使う
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree

dataset = load_iris()
df_iris =  pd.DataFrame(dataset.data,columns=dataset.feature_names) 
# df_iris['target'] = dataset.target  


# #targetの値に応じて0=setosa,1=versicolor,2=virginicaに変換する
# df_iris.loc[df_iris['target'] == 0, 'target'] = "setosa"
# df_iris.loc[df_iris['target'] == 1, 'target'] = "versicolor"
# df_iris.loc[df_iris['target'] == 2, 'target'] = "virginica"


#iris_datasetの標準化
X = df_iris

#-------dataset全体で標準化--------
# StandardScalerで標準化
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

print(X_sc)
#----------------------------------

            # #-------datasetの各列で標準化-------
            # A = df_iris['sepal length (cm)']
            # B = df_iris['sepal width (cm)']
            # C = df_iris['petal length (cm)']
            # D = df_iris['petal width (cm)']

            # # numpyで平均、分散それぞれを計算する。
            # # 平均
            # ave_A=np.average(A)
            # ave_B=np.average(B)
            # ave_C=np.average(C)
            # ave_D=np.average(D)

            # # 分散
            # stdA=np.std(A)
            # stdB=np.std(B)
            # stdC=np.std(C)
            # stdD=np.std(D)

            # # 標準化の線形変換
            # norm_A=[]
            # norm_B=[]
            # norm_C=[]
            # norm_D=[]
            # for i in range(len(A)):
            #     norm_A.append((A[i]-ave_A)/stdA)
            #     norm_B.append((B[i]-ave_B)/stdB)
            #     norm_C.append((C[i]-ave_C)/stdC)
            #     norm_D.append((D[i]-ave_D)/stdD)

            # # A,B,C,Dを結合
            # X_sc = np.c_[A, B, C, D]

            # #-----------------------------------

y = dataset.target

#次元削減
pca = PCA(n_components=2)
pca.fit(X_sc)

# 各主成分によってどの程度カバー出来ているかの割合(第一主成分，第二主成分)
print(pca.explained_variance_ratio_)

# 次元削減をdf_irisに適用する．
feature = pca.fit_transform(X_sc[:, :-1])    #ndarray  最後の列（target）を除いて次元削減
print(feature)
feature_with_target = np.hstack((feature, np.array([y]).T))

fwt = feature_with_target
# カラーマップの色を定義
colors = [  (1, 0, 0),   # 赤
            (0, 1, 0),   # 緑
            (0, 0, 1)]   # 青

# カラーマップを作成
cmap = ListedColormap(colors)

# plt.scatter(fwt[:, 0], fwt[:,1], c=y , cmap=cmap, alpha=0.8, edgecolor='black')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('Standardized PCA of IRIS dataset')

# #カスタム凡例テキストの使用
# patch1 = mpatches.Patch(color='r', label='setosa')
# patch2 = mpatches.Patch(color='g', label='versicolor')
# patch3 = mpatches.Patch(color='b', label='virginica')
# plt.legend(handles=[patch1,patch2,patch3],loc='upper right')
# plt.savefig('Standard_PCA.pdf', format='pdf')       #出力したグラフをiris.pdfという名前で保存する
# plt.show()


#決定木による分類を行う　ここでfeatureは抽出したデータ、fwtは抽出したデータとtargetが入っている。
fwt_train, fwt_test = train_test_split(fwt,test_size=0.3,random_state=42)
    #train_test_split(namelist, test_size=0.3)ここでは30%がテストデータ,random_stateはシード値

#モデル（決定木）を作成
#モデルを定義
clf = tree.DecisionTreeClassifier(max_depth=3)
#モデルの学習
clf.fit(fwt[:,0:2], fwt[:,2])

#作成したモデル木の可視化
plt.figure(figsize=(15,10))
plot_tree(clf, feature_names=fwt[:,2], filled=True)
plt.show()

#作成したモデル（決定木）をもちいて予測を実行
predicted = clf.predict(fwt_test[:,0:2])
print(predicted)

#識別率の確認
print("正解率 =")
print(sum(predicted == fwt_test[:,2])/len(fwt_test[:,2]))

#決定境界の可視化
XX = fwt[:,0:2]     #Principal Component 1, Principal Component 2
YY = fwt[:,2]       #target

#決定境界を描画するためのグリッドを作成
xx_min, xx_max = fwt[:,0].min() - 1, fwt[:,0].max() + 1
yy_min, yy_max = fwt[:,1].min() - 1, fwt[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(xx_min, xx_max, 0.01),
                        np.arange(yy_min, yy_max, 0.01))

# 各グリッドポイントに対して予測を行い、決定境界を計算
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#データを図示する
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
plt.scatter(fwt[:, 0], fwt[:,1], c=y , cmap=cmap, alpha=0.8, edgecolor='black')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Standardized PCA of IRIS dataset')

#カスタム凡例テキストの使用
patch1 = mpatches.Patch(color='r', label='setosa')
patch2 = mpatches.Patch(color='g', label='versicolor')
patch3 = mpatches.Patch(color='b', label='virginica')
plt.legend(handles=[patch1,patch2,patch3],loc='upper right')
plt.savefig('Standard_PCA.pdf', format='pdf')       #出力したグラフをiris.pdfという名前で保存する
plt.show()
