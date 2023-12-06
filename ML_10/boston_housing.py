# boston_housing 住宅価格の推定
import keras 
from keras.datasets import boston_housing

(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# トレーニングデータの正規化
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train = (X_train - X_train_mean) / X_train_std

y_train_mean = y_train.mean(axis=0)
y_train_std = y_train.std(axis=0)
y_train = (y_train - y_train_mean) / y_train_std

#テストデータの正規化
X_test = (X_test - X_train_mean) / X_train_std
y_test = (y_test - y_train_mean) / y_train_std


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1], )))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mse',         # 損失関数　「平均二乗誤差」
    metrics=['mae']     # 評価関数　「平均絶対誤差」
)

history = model.fit(X_train, y_train,   # トレーニングデータ
                    batch_size=1,       # バッチサイズの指定
                    epochs=100,         # エポック数の指定
                    verbose=1,          # ログ出力の指定
                    validation_data=(X_test, y_test))   # テストデータ
