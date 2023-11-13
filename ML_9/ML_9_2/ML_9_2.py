import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#OPTUNAをインポート(OPTUNAはハイパーパラメータの最適化ライブラリ)
import optuna 
#KFoldのインポート
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error
X_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_train_sc.csv")
y_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_train.csv")
X_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_test_sc.csv")
y_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_test.csv")
df_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_train.csv")
df_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_test.csv")

#k分割交差検証を行う
#KFoldの設定
kf = KFold(n_splits = 10, shuffle = True, random_state = 42)

def objective(trial):
    # ハイパーパラメータのサンプリング（多項式の次数を1から10の整数とします）
    degree = trial.suggest_int('degree', 1, 4)
    
    # 多項式特徴量の追加
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    
    # 線形回帰モデルの構築
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # 交差検証で評価
    mse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.loc[train_idx], X_train.loc[val_idx]
        y_train_fold, y_val_fold = y_train.loc[train_idx], y_train.loc[val_idx]
        
        X_train_fold_poly = poly.transform(X_train_fold)
        X_val_fold_poly = poly.transform(X_val_fold)
        
        model_fold = LinearRegression()
        model_fold.fit(X_train_fold_poly, y_train_fold)
        
        y_val_pred = model_fold.predict(X_val_fold_poly)
        mse_fold = mean_squared_error(y_val_fold, y_val_pred)
        mse_scores.append(mse_fold)
    
    # Optunaは最小化の問題を解くので、MSEを返す
    return np.mean(mse_scores)


# Optunaの最適化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

# 結果の表示
best_degree = study.best_params['degree']
best_score = study.best_value

print(f"Best Degree: {best_degree}")
print(f"Best Score: {best_score}")

best_degree = study.best_params['degree']

# 多項式特徴量の追加
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 線形回帰モデルの構築
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 予測値の計算
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# 予測精度の評価（例として、平均二乗誤差を表示）
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"Train Mean Squared Error: {mse_train}")
print(f"Test Mean Squared Error: {mse_test}")

#データに予測値と誤差を追加
df_train['Predict values'] = y_train_pred
df_train['Residuals'] = df_train['Predict values'] - df_train['RoI']
df_test['Predict values'] = y_test_pred
df_test['Residuals'] = df_test['Predict values'] - df_test['RoI']

def get_office_name(case_name):
    name_parts = case_name.split('_',2)  # _で分割
    if len(name_parts) > 0:
        return name_parts[0]
    else:
        return case_name  # 分割できない場合は元の行名を返す

#データをオフィスごとにプロット
df_train['office_name'] = df_train['case_name'].map(get_office_name)

#print(df_train.head())
#print(df_test.head())

plot_column = 'office_name'

# カラーマップを選択
cmap = plt.get_cmap('tab20')

# プロット
fig, ax = plt.subplots(figsize=(8, 6))

unique_values = df_train[plot_column].unique()
colors = cmap(np.linspace(0, 1, len(unique_values)))

for value, color in zip(unique_values, colors):
    subset = df_train[df_train[plot_column] == value]
    ax.scatter(subset['Predict values'], subset['Residuals'], marker = 'o', label = value, color = color)

ax.scatter(df_test['Predict values'], df_test['Residuals'], marker = 'x', label = 'Test data', c = 'gray')
plt.xlabel('Predict values')
plt.ylabel('Residuals')
plt.axhline(y=0, c ='k')
plt.legend()
plt.savefig("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_2/ML_9_2.png")
plt.show()