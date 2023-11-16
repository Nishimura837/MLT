# XGBoostによる勾配ブースティング回帰
import xgboost as xgb
import pandas as pd 
import optuna
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

# 評価指標のインポート
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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

rmses = []
# optunaを使ってｋ回交差検証によりパラメータのチューニングをする
def objective(trial):
    # 調整したいハイパーパラメータについて範囲を指定
    exponentG = trial.suggest_int("exponentG", -20, 10)
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'gamma': 2**exponentG,
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.1),
    }
    

    # KFoldのオブジェクトを作成
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    # KFold CV
    for train_index, valid_index in kf.split(X_train):
        X_train_subset = X_train.loc[train_index]
        y_train_subset = y_train.loc[train_index]
        X_valid_subset = X_train.loc[valid_index]
        y_valid_subset = y_train.loc[valid_index]

        model_X = xgb.XGBRegressor(**param)
        model_X.fit(X_train_subset, y_train_subset)

        y_valid_pred = model_X.predict(X_valid_subset)

        # RMSEを算出
        temp_rmse_valid = np.sqrt(mean_squared_error(y_valid_subset, y_valid_pred))

        # RMSEをリストにappend
        rmses.append(temp_rmse_valid)

        # CVのRMSEの平均値を目的関数として返す
        return np.mean(rmses)
    
folds = 10
# GridSearch----------
search_space = {
    'n_estimators': np.arange(100, 1100, 100),
    'exponentG': np.arange(-20, 10),
    'max_depth': np.arange(3, 8), 
    'subsample': np.arange(0.1, 1.1, 0.1)
    }
study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
study.optimize(objective)
# --------------------  

# # ベイズ最適化--------
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials = 200)
# # --------------------
print('Number of finalized trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

n_estimators = study.best_trial.params['n_estimators']
print('n_estimators', n_estimators)
gamma = 2** study.best_trial.params['exponentG']
print('gamma', gamma)
max_depth = study.best_trial.params['max_depth']
print('max_depth', max_depth)
subsample = study.best_trial.params['subsample']
print('subsample', subsample)

model_X = xgb.XGBRegressor(
    n_estimators=n_estimators,
    gamma=gamma,
    max_depth=max_depth,
    subsample=subsample
)
model_X.fit(X_train, y_train)

y_test_pred = model_X.predict(X_test)
y_train_pred = model_X.predict(X_train)
print('y_test_pred', y_test_pred)
print('R^2 test =', r2_score(y_test, y_test_pred))
print('R^2 train =', r2_score(y_train_pred, y_train))


#各種評価指標をcsvファイルとして出力する
df_ee = pd.DataFrame({'R^2(決定係数)': [r2_score(y_test, y_test_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})
df_ee.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_5/Error Evaluation 9_5 XGB.csv",encoding='utf_8_sig', index=False)

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
# df_forfig.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_5/df_forfig XGB.csv"\
#                         ,encoding='utf_8_sig', index=False)

#-----Error Evaluation (+test) DTR.pdfの作成-------------------------------------------
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


plt.title('Error Evaluation XGB')
plt.savefig("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_5/Error Evaluation (+test) XBG.pdf", format='pdf') 
# plt.show()
#-----------------------------------------------------------------------------------