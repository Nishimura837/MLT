import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# データフレームの作成（実際のデータを使用する場合、データを適切に読み込んでください）
data = pd.DataFrame({
    '説明変数1': [1, 2, 3, 4, 5],
    '説明変数2': [2, 3, 4, 5, 6],
    '目的関数': [3, 4, 5, 6, 7]
})

# 説明変数をXに、目的関数をyに分割
X = data[['説明変数1', '説明変数2']]
y = data['目的関数']

# 定数項（切片）を追加
X = sm.add_constant(X)

# 線形重回帰モデルを作成
model = sm.OLS(y, X)

# モデルをフィット（適合）させる
results = model.fit()

# モデルの統計的な要約を表示
print(results.summary())

# 結果の図示（実際のデータに合わせて適切に調整してください）
plt.scatter(y, results.fittedvalues)
plt.xlabel("実際の目的関数")
plt.ylabel("予測値")
plt.title("線形重回帰分析の結果")
plt.show()
