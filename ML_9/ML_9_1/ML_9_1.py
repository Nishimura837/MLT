# 線形重回帰分析
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# 線形モデル
from sklearn import linear_model
# 評価指標のインポート
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import os

#df_test.csv,df_train.csvを取得
df_test_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/df_test.csv"
df_train_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/df_train.csv"
df_test = pd.read_csv(df_test_path)
df_train = pd.read_csv(df_train_path)

X_train = df_train.drop(columns=['case_name', 'RoI'])
y_train = df_train["RoI"]
X_test = df_test.drop(columns=['case_name', 'RoI'])
y_test = df_test["RoI"]


#重回帰分析では説明変数が複数存在するため、説明変数を標準化する
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# 標準化されたデータを新しいデータフレームに格納
scaled_X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#csvファイルとして出力
scaled_X_train.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/scaled_X_train.csv",encoding='utf_8_sig', index=False)
scaled_X_test.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/scaled_X_test.csv",encoding='utf_8_sig', index=False)

#モデルの作成と適用
model = linear_model.LinearRegression()
model.fit(scaled_X_train, y_train)

#予測の実行
y_test_pred = model.predict(scaled_X_test)
y_train_pred = model.predict(scaled_X_train)

# print('回帰係数')
# print(model.coef_)

#各種評価指標をcsvファイルとして出力する
df_ee = pd.DataFrame({'R^2(決定係数)': [r2_score(y_test, y_test_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})
df_ee.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/Error Evaluation.csv",encoding='utf_8_sig', index=False)
# #決定係数について
# print('R^2(決定係数)')
# print(r2_score(y_test, y_test_pred))
# #RMSE(二乗平均平方根誤差)について
# print('RMSE(二乗平均平方根誤差)')
# print(np.sqrt(mean_squared_error(y_test, y_test_pred)))
# #MSE(平均二乗誤差)について
# print('MSE(平均二乗誤差)')
# print(mean_squared_error(y_test, y_test_pred))
# #MAE(平均絶対誤差)について
# print('MAE(平均絶対誤差)')
# print(mean_absolute_error(y_test, y_test_pred))


# 図を作成するための準備
df_train_forfig = df_train[["case_name", "RoI"]]
df_train_forfig['predict values'] = y_train_pred
df_train_forfig['residuals'] = df_train_forfig['predict values'] - df_train_forfig['RoI']
df_test_forfig = df_test[["case_name", "RoI"]]
df_test_forfig['predict values'] = y_test_pred
df_test_forfig['residuals'] = df_test_forfig['predict values'] - df_test_forfig['RoI']
# print(df_train_forfig)

#'legend'列を追加(凡例)

root_directory = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/"
for folder_name in os.listdir(root_directory):  
    for index,row in df_train_forfig.iterrows() :           #１行ずつ実行
        if folder_name in row['case_name']:                 #case_nameに'folder_nameが含まれているかどうか
            df_train_forfig.loc[index,'legend'] = 'Training:' + folder_name

df_test_forfig['legend'] = 'Test data'

df_forfig = pd.concat([df_train_forfig, df_test_forfig])
df_forfig.to_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/df_forfig.csv"\
                        ,encoding='utf_8_sig', index=False)

#図の作成
# 各オフィス名に対する色を 'tab20' カラーマップから取得
legend_names = df_forfig['legend'].unique()      #unique()メソッドは指定した列内の一意の値の配列を返す（重複を取り除く）
# print(legend_names)
colors = plt.cm.tab20(range(len(legend_names))) #tab20から配列legemd_namesの長さ分の色の配列colorsを返す
# 凡例名と色の対応を辞書に格納
# zip関数は２つ以上のリストを取り、それらの対応する要素をペアにしてイテレータを返す。
#この場合、legend_namesとcolorsの２つのリストをペアにし、対応する要素同士を取得する。
# =以降はofficeをキーとしてそれに対応するcolorが"値"として格納される辞書を作成
legend_color_mapping = {legend: color for legend, color in zip(legend_names, colors)}
# print(legend_color_mapping)
# 'legend' 列を数値（色情報に対応する数値）に変換
# 'legend_num'　を追加
df_forfig['legend_num'] = df_forfig['legend'].map(legend_color_mapping)
#散布図を作成
df_forfig.plot.scatter(x='predict values', y='residuals', c=df_forfig['legend_num'])
#y=0の直線を引く
# y = 0 の直線を描く
plt.axhline(y=0, color='black', linestyle='-')

# 凡例を作成
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, \
                        markersize=10, label=legend) for legend, color in zip(legend_names, colors)]

# 凡例を表示
plt.legend(handles=handles, loc='upper left')


plt.title('Error Evaluation')
plt.savefig("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_1/Error Evaluation.pdf", format='pdf') 
# plt.show()


