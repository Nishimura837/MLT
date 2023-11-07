import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#線形回帰モデル
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_train_sc.csv")
y_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_train.csv")
X_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_test_sc.csv")
y_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_test.csv")
df_train = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_train.csv")
df_test = pd.read_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_test.csv")

#モデルの構築
model = LinearRegression()

#学習
model.fit(X_train, y_train)

#予測
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

#結果
#print("【決定係数(訓練)】:", model.score(X_train, y_train))
#print("【決定係数(テスト)】:", model.score(X_test, y_test))

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
plt.savefig("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_1/ML_9_1.png")
plt.show()
