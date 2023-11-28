import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

X_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_train_sc.csv")
y_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_train.csv")
X_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_test_sc.csv")
y_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_test.csv")
df_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_train.csv")
df_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_test.csv")


kf = KFold(n_splits=10, shuffle=True, random_state=42)


def objective(trial):
    # ハイパーパラメータの探索範囲
    degree = trial.suggest_int('degree', 1, 4)
    alpha = trial.suggest_float('alpha', 0, 20, step=0.1)

    # k分割交差検証の実行
    rmse_scores = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.loc[train_index], X_train.loc[val_index]
        y_train_fold, y_val_fold = y_train.loc[train_index], y_train.loc[val_index]

        poly = PolynomialFeatures(degree=degree, interaction_only=True)
        X_poly_train_fold = poly.fit_transform(X_train_fold)

        lasso = Lasso(alpha=alpha)
        lasso.fit(X_poly_train_fold, y_train_fold)

        X_poly_val_fold = poly.transform(X_val_fold)
        y_pred = lasso.predict(X_poly_val_fold)

        rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        rmse_scores.append(rmse_fold)

    # 平均二乗誤差の平均を目的関数とする
    return np.mean(rmse_scores)


# Optunaで最適なハイパーパラメータの探索
search_space = {'degree': range(1, 5), 'alpha': np.arange(0, 20.1, 0.1)}
study = optuna.create_study(
    sampler=optuna.samplers.GridSampler(search_space), direction='minimize')
study.optimize(objective)

# 結果の表示
print("Best trial:")
trial = study.best_trial
print("  Value: {:.4f}".format(trial.value))
print("  Params: {}".format(trial.params))

# 最も精度の良いモデルで全トレーニングデータを用いて再学習
best_degree = trial.params['degree']
best_alpha = trial.params['alpha']

poly = PolynomialFeatures(degree=best_degree, interaction_only=True)
X_train_poly = poly.fit_transform(X_train)

lasso = Lasso(alpha=best_alpha)
lasso.fit(X_train_poly, y_train)

# テストデータでの予測と評価
X_test_poly = poly.transform(X_test)
y_train_pred = lasso.predict(X_train_poly)
y_test_pred = lasso.predict(X_test_poly)
mse = mean_squared_error(y_test, y_test_pred)
print("Test MSE:", mse)

df_er = pd.DataFrame({'r2(決定係数)': [r2_score(y_test, y_test_pred)],
                     'RMSE(平方平均二乗誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                      'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                      'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})

df_er.to_csv(
    "/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_3/df_er_lasso.csv", index=False)

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
plt.savefig("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_3/ML_9_3_lasso.png")
plt.show()
