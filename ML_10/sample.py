import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam


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

# nan_indices_y_test = np.isnan(y_test)
# print("NaN indices in y_test:", np.where(nan_indices_y_test))

rmses = []
def create_model(trial):
    model = Sequential()
    
    # ハイパーパラメータの最適化
    n_layers = trial.suggest_int('n_layers', 1, 5)
    for i in range(n_layers):
        units = trial.suggest_int(f'n_units_{i}', 8, 64)
        model.add(Dense(units=units, activation='relu'))
    
    # 出力層
    model.add(Dense(units=1, activation='linear'))

    # オプティマイザと学習率の最適化
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=lr)
    else:
        optimizer = 'sgd'

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def objective(trial):
    # モデルの作成
    model = create_model(trial)

    # KFold のオブジェクトを作成
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    
    for train_index, valid_index in kf.split(X_train):
        X_train_subset = X_train.loc[train_index]
        y_train_subset = y_train.loc[train_index]
        X_valid_subset = X_train.loc[valid_index]
        y_valid_subset = y_train.loc[valid_index]

        X_train_subset = X_train.values
        y_train_subset = y_train.values
        X_valid_subset = X_test.values
        y_valid_subset = y_test.values  


        # トレーニング
        model.fit(X_train_subset, y_train_subset, epochs=10, batch_size=16, verbose=0)

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

folds = 10

# Optunaの最適化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# 最適なハイパーパラメータの表示
print('Best trial:')
trial = study.best_trial
print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')

# 最適なモデルの作成
best_model = create_model(trial)

X_test = X_test.values
y_test = y_test.values
# テストデータでの評価
y_pred = best_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_pred)
print(f'Final MSE on test data: {final_mse}')
print(y_pred)

# 残差プロットの作成
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()
