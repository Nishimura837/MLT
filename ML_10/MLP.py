import numpy as np
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.optimizers import Adam

# ①入力を定義
# 3次元の入力
# 正解ラベルは0か1
X = np.array([
    [0.3, 0.2, -1.1],
    [0.2, 0.6, -0.9],
    [0.4, 0.3, -0.3],
])

y = np.array([0, 1, 0])

# ②インスタンス
model = Sequential()

# ③中間層(1層目)作成
# input=dim(入力の次元):3次元
# units(パーセプトロン数):2つ
# activation(活性化関数):ReLU関数

model.add(Dense(units=2, activation="relu", input_dim=3))

# ④中間層(2層目)作成
# input_dim(入力の次元):3次元
# units(パーセプトロン数):2つ
# activation(活性化関数):ReLU関数

model.add(Dense(units=2, activation="relu", input_dim=3))

# ⑤出力層追加
# input_dim(入力の次元):省略(前の層のperceptron数が2つだから自動的に2次元に決定する)
# units(パーセプトロン数):1つ
# activation(活性化関数):sigmoid関数
# 出力は1次元

model.add(Dense(units=1, activation="sigmoid"))

# ⑥学習方法決定
# loss(損失関数):binary_crossentropy(2分類問題で利用)
# 損失関数を最小化するためのパラメータ最適化アルゴリズム(optimizer):adam
# 学習率(lr):0.1

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.1))

# ⑦学習
history = model.fit(X, y, epochs=1000)


# 層を指定し、重みとバイアスの値を確認
# 重み
print(model.layers[0].get_weights()[0])
# バイアス
print(model.layers[0].get_weights()[1])


# 損失関数の誤差の変化を確認する
# フォントの日本語対応

loss = history.history['loss']

plt.plot(np.arange(len(loss)), loss, label="train error")
plt.legend()
plt.show()