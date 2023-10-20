from sklearn.datasets import load_iris

# アヤメのデータセットを読み込む
iris = load_iris()

# データの特徴量（説明変数）を取得
X = iris.data

# データのクラスラベル（目的変数）を取得
y = iris.target

# データの特徴量の説明
print("特徴量（説明変数）:")
print(iris.feature_names)

# データのクラスラベルの説明
print("\nクラスラベル（目的変数）:")
print(iris.target_names)
