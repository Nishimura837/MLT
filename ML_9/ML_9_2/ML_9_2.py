# 多項式回帰分析
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
# 評価指標のインポート
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#df_test.csv,df_train.csvを取得
df_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test.csv"
df_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train.csv"
df_test = pd.read_csv(df_test_path)
df_train = pd.read_csv(df_train_path)

X_train = df_train.drop(columns=['case_name', 'RoI'])
y_train = df_train["RoI"]
X_test = df_test.drop(columns=['case_name', 'RoI'])
y_test = df_test["RoI"]


#説明変数が複数存在するため、説明変数を標準化する
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# 標準化されたデータを新しいデータフレームに格納
scaled_X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#k分割交差検証によってハイパーパラメータ(degree)を決定する
print('start cross_val')
degrees = list(range(1, 4))  # 1から3の多項式次数を試す

best_degree = None
best_score = -np.inf

for degree in degrees:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    scores = cross_val_score(model, scaled_X_train , y_train, cv=10, scoring='r2')  # 10分割交差検証でR-squaredスコアを計算
    mean_score = scores.mean()  # 10分割の平均スコアを計算

    print(degree)

    if mean_score > best_score:
        best_score = mean_score
        best_degree = degree

print('end cross_val')
print(best_degree)

# 多項式回帰を行う
polynomial_features= PolynomialFeatures(degree=best_degree)
x_train_poly = polynomial_features.fit_transform(scaled_X_train)
x_test_poly = polynomial_features.transform(scaled_X_test)


# y = b0 + b1x + b2x^2 の b0～b2 を算出
model = LinearRegression()
model.fit(x_train_poly, y_train)
y_train_pred = model.predict(x_train_poly)
y_test_pred = model.predict(x_test_poly)


#各種評価指標をcsvファイルとして出力する
df_ee = pd.DataFrame({'R^2(決定係数)': [r2_score(y_test, y_test_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})
df_ee.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_2/Error Evaluation 9_2.csv",encoding='utf_8_sig', index=False)

