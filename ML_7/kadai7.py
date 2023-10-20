import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression #ロジスティック回帰の読み込み

# Irisデータセットの読み込み
iris = load_iris()
X = iris.data
y = iris.target

# SetosaとVersicolorを1つのクラス、Virginicaをもう1つのクラスに分類
y_binary = np.where(y == 0, 0, 1)  # SetosaとVersicolorを0に、Virginicaを1に設定

# データをトレーニングデータとテストデータに分割（7:3の割合）
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# ロジスティック回帰モデルを作成し、トレーニングデータで学習
model = LogisticRegression()
model.fit(X_train, y_train)

# テストデータで予測
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc #混同行列とF1スコアとAUCの導入
from sklearn.preprocessing import label_binarize


# 混同行列を計算
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

#混同行列の可視化
import seaborn as sns
import pandas as pd
conf_matrix = pd.DataFrame(data=conf_matrix, index=["act_setosa & versicolor", "act_virginica"], 
                           columns=["pred_setosa & versicolor", "pred_virginica"])
sns.heatmap(conf_matrix, square=True, cbar=True, annot=True, cmap='Blues')
plt.yticks(rotation=90)

# F1スコアを計算
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

# クラスの確率を予測
y_prob = model.predict_proba(X_test)

# Virginicaの確率を取得
virginica_prob = y_prob[:, 1]

# ROC曲線を計算
fpr, tpr, thresholds = roc_curve(y_test, virginica_prob)

# AUCを計算
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

# ROC曲線をプロット
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='b', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], lw=2, color = 'black', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()
