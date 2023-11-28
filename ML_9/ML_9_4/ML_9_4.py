import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from optuna.samplers import GridSampler
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# データの読み込みや前処理が必要な場合はここで行う
X_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_train_sc.csv")
y_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_train.csv")
X_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_test_sc.csv")
y_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_test.csv")
df_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_train.csv")
df_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_test.csv")


# k分割交差検証のためのKFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)


def objective(trial):
    # ハイパーパラメータのサンプリング
    C = trial.suggest_float('C', 0.1, 1.0, step=0.1)
    gamma = trial.suggest_float('gamma', 0.1, 1.0, step=0.1)
    epsilon = trial.suggest_float('epsilon', 0.1, 1.0, step=0.1)

    # SVRモデルの構築
    svr = SVR(C=C, gamma=gamma, epsilon=epsilon)

    # 交差検証の実行
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.loc[train_idx], X_train.loc[val_idx]
        y_fold_train, y_fold_val = y_train.loc[train_idx], y_train.loc[val_idx]

        svr.fit(X_fold_train, y_fold_train)
        y_pred = svr.predict(X_fold_val)
        rmse_fold = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        rmse_scores.append(rmse_fold)

    # 目的関数（最小化したい指標）の計算
    rmse = np.mean(rmse_scores)
    return rmse


# Optunaによるハイパーパラメータの最適化(GridSamplerを使用)
sampler = optuna.samplers.GridSampler(
    {'C': np.arange(0.1, 1, 10), 'gamma': np.arange(0.01, 0.1, 1), 'epsilon': np.arange(0.01, 0.1, 1)})
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective)

# 最適なハイパーパラメータでのSVRモデルのトレーニング
best_params = study.best_params
best_svr = SVR(**best_params)
best_svr.fit(X_train, y_train)

# テストデータに対する予測
y_test_pred = best_svr.predict(X_test)
y_train_pred = best_svr.predict(X_train)

# モデルの評価
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Best Hyperparameters: {best_params}')

df_er = pd.DataFrame({'r2(決定係数)': [r2_score(y_test, y_test_pred)],
                     'RMSE(平方平均二乗誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                      'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                      'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})

df_er.to_csv(
    "/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_4/df_er.csv", index=False)

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
plt.title('lasso')
plt.savefig("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_4/ML_9_4.png")
plt.show()
