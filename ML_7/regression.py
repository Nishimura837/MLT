import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# Irisデータセットをロード
iris = load_iris()
X = iris.data
y = iris.target

# setosaとversicolorのデータを選択
X_selected = X[y != 2]
y_selected = y[y != 2]

# トレーニングデータとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=0.3, random_state=42)

# ロジスティック回帰モデルを作成
model = LogisticRegression()

# モデルをトレーニング
model.fit(X_train, y_train)

# 予測を行う
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 混合行列を計算
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)

# F1スコアを計算
f1_score_train = f1_score(y_train, y_train_pred)
f1_score_test = f1_score(y_test, y_test_pred)

# ROC曲線とAUCを計算
y_train_prob = model.predict_proba(X_train)[:, 1]
y_test_prob = model.predict_proba(X_test)[:, 1]
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

# 結果を出力
print("Training Confusion Matrix:")
print(confusion_matrix_train)
print("Training F1 Score:", f1_score_train)
print("Training AUC:", auc_train)

print("\nTest Confusion Matrix:")
print(confusion_matrix_test)
print("Test F1 Score:", f1_score_test)
print("Test AUC:", auc_test)

# ROC曲線をプロット
plt.figure()
#plt.plot(fpr_train, tpr_train, color='darkorange', label='Training ROC curve (area = %0.2f)' % auc_train)
#plt.plot(fpr_test, tpr_test, color='cornflowerblue', label='ROC curve (area = %0.2f)' % auc_test)
plt.plot(fpr_test, tpr_test, color='cornflowerblue', label='ROC curve' % auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()