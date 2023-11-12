# 多項式回帰分析、k分割交差検証、optunaを使用
import pandas as pd
import optuna
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import os
# 評価指標のインポート
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#test.csv,train.csvを取得
X_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_train.csv"
y_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_train.csv"
X_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_test.csv"
y_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_test.csv"
df_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train.csv"
df_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test.csv"
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)
df_train = pd.read_csv(df_train_path)
df_test = pd.read_csv(df_test_path)

# optunaを使ってk分割交差検証を行う-------------------------
def objective(trial):
    # 調整したいハイパーパラメータについて範囲を指定
    param = {'degree':trial.suggest_int('degree', 1, 4)}

    # 学習に使用するアルゴリズムを指定
    model_P = PolynomialFeatures(**param, interaction_only=True)

    # データを変換
    Poly_X = model_P.fit_transform(X_train)

    # k分割交差検証を実行
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(LinearRegression(), Poly_X, y_train, cv=kf, scoring='neg_mean_squared_error')

    # 評価指標の平均を計算
    score = -np.mean(scores)    # 平均二乗誤差を最小化するためにマイナスをかけて符の値で出てきている平均二乗誤差を正の値に変換している
                                # 基本的には最大化問題であるため、平均二乗誤差を符の値で算出するが、optunaは最小化問題を解くことを前提としているため、
                                # マイナスをかけることで最小化問題にしている
    return score

study = optuna.create_study(direction='minimize')   # 最小化問題として設定
study.optimize(objective, n_trials=30)              # 試行回数を調整

best_degree = study.best_params['degree']
#---------------------------------------------------------

print('Number of finalized trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
degree = list(study.best_trial.params.values())[0]
print('degree:', degree)
# 最適なハイパーパラメータを設定したモデルの定義
best_model = PolynomialFeatures(degree=degree, interaction_only=True)
best_model.fit(X_train)
X_train_poly = best_model.fit_transform(X_train)
X_test_poly = best_model.transform(X_test)


#モデルの適合
model = LinearRegression()
model.fit(X_train_poly, y_train)
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)
print('y_train_pred', y_train_pred)
print('y_test_pred', y_test_pred)

#各種評価指標をcsvファイルとして出力する
df_ee = pd.DataFrame({'R^2(決定係数)': [r2_score(y_test, y_test_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})
df_ee.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_2/Error Evaluation 9_2 sub.csv",encoding='utf_8_sig', index=False)

df_ee_train = pd.DataFrame({'R^2(決定係数)': [r2_score(y_train, y_train_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y_train, y_train_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y_train, y_train_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y_train, y_train_pred)]})
df_ee_train.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_2/Error Evaluation for traindata 9_2 sub.csv",encoding='utf_8_sig', index=False)

# 図を作成するための準備
df_train['predict values'] = y_train_pred
df_train['residuals'] = df_train['predict values'] - df_train['RoI']
df_test['predict values'] = y_test_pred
df_test['residuals'] = df_test['predict values'] - df_test['RoI']


#df_trainに'legend'列を追加(凡例)
root_directory = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/"
for folder_name in os.listdir(root_directory):  
        for index,row in df_train.iterrows() :           #１行ずつ実行
                if folder_name + '_' in row['case_name']:                 #case_nameにfolder_nameが含まれているかどうか
                        df_train.loc[index,'legend'] = 'Training:' + folder_name

df_test['legend'] = 'Test data'

df_forfig = pd.concat([df_train, df_test])
df_forfig.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_1/df_forfig_sub.csv"\
                        ,encoding='utf_8_sig', index=False)

#図の作成
# 各オフィス名に対する色を 'tab20' カラーマップから取得
legend_names = df_train['legend'].unique()      #unique()メソッドは指定した列内の一意の値の配列を返す（重複を取り除く）
# print(legend_names)
colors = plt.cm.tab20(range(len(legend_names))) #tab20から配列legemd_namesの長さ分の色の配列colorsを返す
# 凡例名と色の対応を辞書に格納
# zip関数は２つ以上のリストを取り、それらの対応する要素をペアにしてイテレータを返す。
# この場合、legend_namesとcolorsの２つのリストをペアにし、対応する要素同士を取得する。
# =以降はofficeをキーとしてそれに対応するcolorが"値"として格納される辞書を作成
legend_color_mapping = {legend: color for legend, color in zip(legend_names, colors)}
# print(legend_color_mapping)
# 'legend' 列を数値（色情報に対応する数値）に変換
# 'legend_num'　を追加
df_train['legend_num'] = df_train['legend'].map(legend_color_mapping)
#散布図を作成
plt.scatter(df_train['predict values'], df_train['residuals'], c=df_train['legend_num'])
plt.scatter(df_test['predict values'], df_test['residuals'], c='black', marker='x' )
#y=0の直線を引く
# y = 0 の直線を描く
plt.axhline(y=0, color='black', linestyle='-')

# 凡例を作成
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, \
                        markersize=6, label=legend) for legend, color in zip(legend_names, colors)]
# Test dataの凡例を追加
handles[-1] = plt.Line2D([0], [0] ,marker='x', color='black', markersize=6, label='Test data', linestyle='None')

# 凡例を表示
plt.legend(handles=handles, loc='upper left', fontsize=6)


plt.title('Error Evaluation optuna')
plt.savefig("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_2/Error Evaluation (+test) optuna sub.pdf", format='pdf') 
# plt.show()