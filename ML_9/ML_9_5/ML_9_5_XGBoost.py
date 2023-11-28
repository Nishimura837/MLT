import optuna
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_train_sc.csv")
y_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_train.csv")
X_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_test_sc.csv")
y_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_test.csv")
df_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_train.csv")
df_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_test.csv")

# ハイパーパラメータの最適化を行う関数


def objective(trial):
    # ハイパーパラメータの探索範囲を指定
    params = {
        'booster': 'gbtree',
        'lambda': trial.suggest_float('lambda', 0.1, 1.0),  # L2正則化項
        'max_depth': trial.suggest_int('max_depth', 1, 10),   # 最大深さ
        'eta': trial.suggest_float('eta', 0.1, 1.0),   # 学習率
        'gamma': trial.suggest_float('gamma', 0.1, 1.0),  # 損失減少の下限
        'n_estimators': trial.suggest_int('n_estimators', 10, 100, step=10)
    }

    # モデルの定義
    model = xgb.XGBRegressor(**params)

    # 10-foldの交差検証を手動で実行
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    rmse_scores = []
    for train_idx, valid_idx in kf.split(X_train):
        X_train_fold, X_valid_fold = X_train.loc[train_idx], X_train.loc[valid_idx]
        y_train_fold, y_valid_fold = y_train.loc[train_idx], y_train.loc[valid_idx]

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_valid_fold)

        rmse_fold = mean_squared_error(
            y_valid_fold, y_pred_fold, squared=False)
        rmse_scores.append(rmse_fold)

    # 交差検証の結果を最小化するように設定
    return np.mean(rmse_scores)


# Optunaでハイパーパラメータの最適化を実行
sampler = TPESampler(seed=42)
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=100)

# 最適なハイパーパラメータを取得
best_params = study.best_trial.params
print("Best Params:", best_params)

# 最適なパラメータでモデルを学習
best_model = xgb.XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

# テストデータで予測
y_test_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)

# テストデータの評価
test_rmse = mean_squared_error(y_test, y_test_pred)
train_rmse = mean_squared_error(y_train, y_train_pred)
print("Test RMSE:", test_rmse)
print("Train RMSE:", train_rmse)

df_er = pd.DataFrame({'r2(決定係数)': [r2_score(y_test, y_test_pred)],
                     'RMSE(平方平均二乗誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                      'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                      'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})

df_er.to_csv(
    "/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_5/df_er_XGBoost.csv", index=False)

# データに予測値と誤差を追加
df_train['Predict values'] = y_train_pred
df_train['Residuals'] = df_train['Predict values'] - df_train['RoI']
df_test['Predict values'] = y_test_pred
df_test['Residuals'] = df_test['Predict values'] - df_test['RoI']


def get_office_name(case_name):
    name_parts = case_name.split('_', 2)  # _で分割
    if len(name_parts) > 0:
        return name_parts[0]
    else:
        return case_name  # 分割できない場合は元の行名を返す


# データをオフィスごとにプロット
df_train['office_name'] = df_train['case_name'].map(get_office_name)

# print(df_train.head())
# print(df_test.head())

plot_column = 'office_name'

# カラーマップを選択
cmap = plt.get_cmap('tab20')

# プロット
fig, ax = plt.subplots(figsize=(8, 6))

unique_values = df_train[plot_column].unique()
colors = cmap(np.linspace(0, 1, len(unique_values)))

for value, color in zip(unique_values, colors):
    subset = df_train[df_train[plot_column] == value]
    ax.scatter(subset['Predict values'], subset['Residuals'],
               marker='o', label=value, color=color)

ax.scatter(df_test['Predict values'], df_test['Residuals'],
           marker='x', label='Test data', c='gray')
plt.xlabel('Predict values')
plt.ylabel('Residuals')
plt.axhline(y=0, c='k')
plt.legend()
plt.title('XGboost')
plt.savefig("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_5/ML_9_5_XGBoost.png")
plt.show()
