# ML_10_trial_optuna.py に加えて、
# バッチサイズの最適化、エポック数は収束したら止めるようにする。

import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from optuna.samplers import TPESampler, GridSampler


#test.csv,train.csvを取得
X_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_train.csv"
y_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_train.csv"
X_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_test.csv"
y_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_test.csv"
df_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train.csv"
df_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test.csv"

# Numpy配列としてよみこむ
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)


rmses = []
folds = 10

def create_model(trial):
    model = Sequential()

    # 入力層を作成
    model.add(Dense(25, activation='relu', input_dim=25))

    # 中間層の作成
    # ハイパーパラメータの最適化
    n_layers = trial.suggest_int('n_layers', 1, 5)
    units = trial.suggest_int('n_units', 8, 256, step=8)
    for i in range(n_layers):
        model.add(Dense(units=units, activation='relu'))

    # 出力層の作成
    model.add(Dense(1))

    # オプティマイザと学習率の最適化
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'RMSProp'])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)

    if optimizer_name == 'adam':
        optimizer = Adam()
    if optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = 'RMSProp'

    model.compile(optimizer=optimizer, loss='mean_squared_error')

    return model

def objective(trial):
    # モデルの作成
    model = create_model(trial)

    # batch_size範囲を指定
    batch_size = trial.suggest_int('batch_size', 16, 256, step=16)

    # KFold のオブジェクトを作成
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    
    for train_index, valid_index in kf.split(X_train):
        X_train_subset = X_train.loc[train_index].values
        y_train_subset = y_train.loc[train_index].values
        X_valid_subset = X_train.loc[valid_index].values
        y_valid_subset = y_train.loc[valid_index].values

        # トレーニング
        model.fit(
            X_train_subset, y_train_subset, 
            batch_size=batch_size,
            epochs=100, 
            verbose=1
        )

        # 予測
        y_pred = model.predict(X_valid_subset)

        # NaN が含まれているか確認
        if np.isnan(y_pred).any():
            print(" NaN が含まれているので関数から抜け出します。")
            return 

        # RMSEを算出
        temp_rmse_valid = np.sqrt(mean_squared_error(y_valid_subset, y_pred))

        # RMSEをリストにappend
        rmses.append(temp_rmse_valid)

        # CVのRMSEの平均値を目的関数として返す
        return np.mean(rmses)
    

# ------------------------------------
sampler = TPESampler()
study = optuna.create_study(sampler=sampler, direction='minimize')
study.optimize(objective, n_trials=50)
# # ------------------------------------

# ------------------------------------
# search_space = {
#         'n_layers' : np.arange(1, 6),
#         'n_units': np.arange(8, 264, 8),
#         'optimizer': np.array(['adam', 'sgd', 'RMSProp']),
#         'lr': np.logspace(start=-5, stop=-1, num=5),
#         'batch_size': np.arange(16, 272, 16),  
#         }
# study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
# study.optimize(objective)
# ------------------------------------

print('Number of finalized trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

# 最適なパラメータの表示

for key, value in study.best_trial.params.items():
    print(f'    {key}: {value}')


X_train = X_train.values
y_train = y_train.values
X_test = X_test.values
y_test = y_test.values

best_batch_size = study.best_trial.params['batch_size']
print(best_batch_size)
# 最適なパラメータを使ってモデルの作成
best_model = create_model(study.best_trial)
train_history = best_model.fit(
                                X_train, y_train, 
                                batch_size=best_batch_size,
                                epochs=100, 
                                verbose=0
                            )

print(train_history.history.keys())

print(train_history.history)
print(len(train_history.history['loss']))

# エポックごとの損失関数値をプロットしてみる
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(train_history.history['loss'])
ax.set_xlabel('Epoch')
ax.set_ylabel('loss')
plt.show()

# テストデータでの評価
y_test_pred = best_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_test_pred)
print('Final MSE on test data:', final_mse)
# print(y_test_pred)

# 横軸に予測値、縦軸に残差をとった図を作成する
residuals = y_test_pred - y_test
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(y_test_pred, residuals)
# y = 0 の直線を描く
plt.axhline(y=0, color='black', linestyle='-')
ax.set_xlabel('y_pred')
ax.set_ylabel('residuals')
plt.title('Error Evaluation 10 trial op2')
plt.savefig("/home/gakubu/デスクトップ/ML_git/MLT/ML_10/Error Evaluation 10 trial op2.pdf", format='pdf') 

#各種評価指標をcsvファイルとして出力する
df_ee = pd.DataFrame({'R^2(決定係数)': [r2_score(y_test, y_test_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})
df_ee.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_10/Error Evaluation 10 trial op2.csv",encoding='utf_8_sig', index=False)