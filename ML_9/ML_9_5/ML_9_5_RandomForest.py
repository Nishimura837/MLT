import optuna
from sklearn.ensemble import RandomForestRegressor
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


kf = KFold(n_splits=10, shuffle=True, random_state=42)


def objective(trial):
    # ハイパーパラメータの探索範囲
    n_estimators = trial.suggest_int('n_estimators', 10, 1000, step=10)
    max_depth = trial.suggest_int('max_depth', 1, 10)

    # クロスバリデーションによる評価
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.loc[train_idx], X_train.loc[val_idx]
        y_train_fold, y_val_fold = y_train.loc[train_idx], y_train.loc[val_idx]

        # 決定木回帰モデルの定義
        rf_regressor = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, criterion='squared_error', random_state=42)
        # モデルの学習
        rf_regressor.fit(X_train_fold, y_train_fold)

        # バリデーションデータに対する予測
        y_val_pred = rf_regressor.predict(X_val_fold)

        # 平均二乗誤差を評価指標とする
        rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_val_pred))
        rmse_scores.append(rmse_fold)

    return np.mean(rmse_scores)


# Optunaによる最適化
sampler = TPESampler(seed=42)  # Use TPESampler instead of GridSampler
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=30)


# 最適なハイパーパラメータの表示
best_params = study.best_trial.params
print("Best Parameters: ", best_params)

# 最も制度の良いモデルの取得
best_regressor = RandomForestRegressor(
    n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], criterion='squared_error', random_state=42)

# 全トレーニングデータを用いたモデルの学習
best_regressor.fit(X_train, y_train)

# テストデータに対する予測
y_test_pred = best_regressor.predict(X_test)
y_train_pred = best_regressor.predict(X_train)
# テストデータに対する予測精度の評価
mse_test = mean_squared_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error on Test Data: ", mse_test)
print("Mean Squared Error on Train Data: ", mse_train)

df_er = pd.DataFrame({'r2(決定係数)': [r2_score(y_test, y_test_pred)],
                     'RMSE(平方平均二乗誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                      'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                      'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})

df_er.to_csv(
    "/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_5/df_er_RandomForest.csv", index=False)

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
plt.title('RandomForest')
plt.savefig("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_5/ML_9_5_RandomForest.png")
plt.show()
