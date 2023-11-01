#office１つで重回帰分析をやってみる
import pandas as pd
import matplotlib.pyplot as plt
# 線形モデル
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

X = pd.read_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/office2/inputdata.csv")
y = pd.read_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/office2/outputdata.csv")
X = X.drop(columns=["case_name", "exhaust", "aircon_position_x", "aircon_position_y"])
y = y["RoI"]
print("X")
print(X)
print("y")
print(y)



model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)

df = pd.DataFrame({'R^2(決定係数)': [r2_score(y, y_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y, y_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y, y_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y, y_pred)]})
plt.scatter(y,y_pred-y)
# y = 0 の直線を描く
plt.axhline(y=0, color='black', linestyle='-')
plt.show()
