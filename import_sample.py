import pandas as pd                                     #pandasをpdとしてimport
import numpy as np                                      #numpyをnpとしてimport
from sklearn.datasets import load_iris                  #load_irisをimport
from sklearn.model_selection import train_test_split    #トレーニングデータとテストデータに分ける
from sklearn import tree                                #決定木をimport
import matplotlib.pyplot as plt                         #pyplotをpltとしてimport
from sklearn.tree import plot_tree                      #決定木を可視化する
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches                   #カスタム凡例テキストを使用するためにpatchesを使う
from sklearn.decomposition import PCA                   #主成分分析(PCA)をimport

dataset = load_iris() 
df_iris =  pd.DataFrame(dataset.data,columns=dataset.feature_names) 
df_iris['target'] = dataset.target
print(df_iris)
