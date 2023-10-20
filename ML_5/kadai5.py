import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

#Irisデータをsklearnから読み込み
iris = load_iris()


#辞書型に近い型でデータが入っているので必要箇所を指定してdataとcolumnsとしてdfで読み込み
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

feature = iris.data
target = iris.target

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
iris_df = pca.fit_transform(iris_df)

plt.figure(figsize=(8, 8))
plt.scatter(iris_df[1:50, 0], iris_df[1:50, 1], marker = 'o', c = 'r', label = 'Setosa')
plt.scatter(iris_df[51:100, 0], iris_df[51:100, 1], marker = 'x', c = 'g', label = 'Versicolor')
plt.scatter(iris_df[101:150, 0], iris_df[101:150, 1], marker = 's', c = 'b', label = 'Varginica')
plt.legend(loc = 'best')
plt.show()
