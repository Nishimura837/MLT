import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.layers import Dense
from keras.models import Sequential


#test.csv,train.csvを取得
X_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_train.csv"
y_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_train.csv"
X_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_test.csv"
y_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_test.csv"
df_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train.csv"
df_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test.csv"

# Numpy配列としてよみこむ
X_train = pd.read_csv(X_train_path).values
y_train = pd.read_csv(y_train_path).values
X_test = pd.read_csv(X_test_path).values
y_test = pd.read_csv(y_test_path).values


# データのサイズを確認
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

# モデルの作成
# 今回のデータは25次元のデータであるから入力層のニューロン数は25
# 出力層にはyの値を予測したいからニューロン数が1
# とりあえず2つの中間層を置き、それぞれニューロン数を64 , 32 とする
# 活性化関数は中間層にはReLU関数を、出力層にはlinearを指定
nn1 = 150
model = Sequential()
model.add(Dense(25, activation='relu', input_dim=25))
model.add(Dense(nn1, activation='relu'))
model.add(Dense(1, activation='linear'))

# モデルをコンパイルする
# 損失関数には平均事情誤差を使い、最適化アルゴリズムはRMSPropオプティマイザを指定
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# トレーニングデータを用いてモデルを学習させる
# ミニバッチサイズとエポック数は検討が必要
# 途中経過はtrain_historyに格納される
train_history = model.fit(
    X_train, y_train,
    batch_size=100,
    epochs=500,
    verbose=1
)

print(train_history.history.keys())

print(train_history.history)
print(len(train_history.history['loss']))

# エポックごとの損失関数値をプロットしてみる
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(train_history.history['loss'])
ax.set_xlabel('Epoch')
ax.set_ylabel('loss')
plt.show()


# 予測とその評価
y_pred = model.predict(X_test)
print(y_pred)

# 横軸に予測値、縦軸に残差をとった図を作成する
residuals = y_pred - y_test
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(y_pred, residuals)
# y = 0 の直線を描く
plt.axhline(y=0, color='black', linestyle='-')
ax.set_xlabel('y_pred')
ax.set_ylabel('residuals')
plt.title('Error Evaluation 10 trial')
plt.savefig("/home/gakubu/デスクトップ/ML_git/MLT/ML_10/Error Evaluation 10 trial.pdf", format='pdf') 



#各種評価指標をcsvファイルとして出力する
df_ee = pd.DataFrame({'R^2(決定係数)': [r2_score(y_test, y_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y_test, y_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_pred)]})
df_ee.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_10/Error Evaluation 10 trial.csv",encoding='utf_8_sig', index=False)