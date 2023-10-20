import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Irisデータセットを読み込みます
iris = load_iris()
X = iris.data
y = iris.target

# SetosaとVersicolorを1つのクラス、Virginicaをもう1つのクラスに分類します
y_binary = np.where(y == 0, 0, 1)  # SetosaとVersicolorを0に、Virginicaを1に設定

# データをトレーニングデータとテストデータに分割します（7:3の割合）
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# ロジスティック回帰モデルを作成し、トレーニングデータで学習させます
model = LogisticRegression()
model.fit(X_train, y_train)

# テストデータで予測を行います
y_pred = model.predict(X_test)

# 混同行列を計算します
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# F1スコアを計算します
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# クラスの確率を予測します
y_prob = model.predict_proba(X_test)

# Virginicaの確率を取得します
virginica_prob = y_prob[:, 1]

# ROC曲線を計算します
fpr, tpr, thresholds = roc_curve(y_test, virginica_prob)

# AUCを計算します
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

# ROC曲線をプロットします
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
#このプログラムは、SetosaとVersicolorを合わせた1つのクラスとVirginicaをもう1つのクラスとして2クラス分類を行い、混同行列、F1スコア、ROC曲線、AUCを計算し、ROC曲線をプロットします。データの分割やモデルのパラメータを調整することで、評価結果を改善することができます。





