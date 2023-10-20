import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# サンプルデータフレームの作成
data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [2, 3, 4, 5, 6],
    'Feature3': [3, 4, 5, 6, 7],
    'Target': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)

# 目的変数（Target）を選択
y = df['Target']

# 説明変数の列名のリスト
features = ['Feature1', 'Feature2', 'Feature3']

# ヒートマップを作成するためのループ
for i in range(len(features)):
    for j in range(i + 1, len(features)):
        # 2つの説明変数を選択
        selected_features = [features[i], features[j]]
        X = df[selected_features]

        # 相関行列を計算
        correlation_matrix = X.corr()

        # ヒートマップを描画
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title(f"Correlation Heatmap for {selected_features}")
        plt.show()
