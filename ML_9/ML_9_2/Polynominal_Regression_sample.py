import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# 適当な説明変数を生成
X = np.random.rand(100, 1) * 10
# 適当な目的関数を生成（例：y = 2x^2 - 3x + 1 + ノイズ）
y = 2 * X**2 - 3 * X + 1 + np.random.randn(100, 1)

# 多項式回帰の次数を設定
degree = 2

# 多項式回帰モデルを作成
model = make_pipeline(PolynomialFeatures(degree, interaction_only=True), LinearRegression())
model.fit(X, y)

# 予測値を計算
y_pred = model.predict(X)

# 残差を計算
residuals = y - y_pred

# 予測値と残差の散布図をプロット
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, c='b', marker='o', label='Residuals')
plt.axhline(y=0, color='r', linestyle='--', label='Zero Residual Line')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.legend()
plt.title('Residual Plot in Polynomial Regression')
plt.grid()
plt.show()
