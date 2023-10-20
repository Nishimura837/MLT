import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#データセットを読み込む
dataset = load_iris() 
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['target'] = dataset.target
print(df)
print(dataset['target_names'])

#targetの値に応じて0と1を0に、2を1に変換することでsetosaとversicolorを１つのクラスとする
df.loc[df['target'] == 0, 'target'] = 0
df.loc[df['target'] == 1, 'target'] = 0
df.loc[df['target'] == 2, 'target'] = 1

df_merged = df

dfm = df_merged

#ロジスティック回帰を用いて分類を行う
X = dfm.iloc[:,0:4]      #説明変数
Y = dfm.iloc[:,4]        #目的変数

#トレーニングデータとテストデータをに分ける
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
print(X_test)

lr = LogisticRegression(random_state=42)   #ロジスティック回帰モデルのインスタンスを作成
lr.fit(X_train, Y_train)    #ロジスティック回帰モデルの重みを学習

print("coefficient =", lr.coef_)
print("intercept =", lr.intercept_)

Y_pred = lr.predict(X_test)
print(Y_pred)

#混同行列を表示する
cm = confusion_matrix(y_pred=Y_pred, y_true=Y_test)
cmp = ConfusionMatrixDisplay(cm, display_labels=["pred_setosa & versicolor",\
                "pred_virginica"])
cmp.plot(cmap=plt.cm.Reds)
plt.show()

#f1 scoreを確認する
print('f1 score =', f1_score(y_true=Y_test, y_pred=Y_pred))

#ROC曲線
Y_score = lr.predict_proba(X_test)[:, 1]        #テストデータがクラス1に属する確率
fpr, tpr, thresholds = roc_curve(y_true=Y_test, y_score=Y_score)    #偽陽性率(FPR),真陽性率(TPR),閾値(thresholds)を返す

plt.plot(fpr, tpr, label='roc curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='random', c='black')            #ランダムな予測を表す対角線をプロットする
plt.plot([0, 0, 1], [0, 1, 1], linestyle='--', label='ideal', c='green')       #理想的な予測を表す線をプロットする
plt.legend()
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC Curve')
plt.show()

#AUCを確認する
print('AUC =', roc_auc_score(y_true=Y_test, y_score=Y_score))
