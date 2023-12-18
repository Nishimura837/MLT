import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.preprocessing .image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import tensorflow as tf
import datetime
import optuna
from sklearn.model_selection import KFold
from optuna.samplers import TPESampler, GridSampler
from sklearn.manifold import TSNE

# 最適化の分割数と試行回数、エポック数
folds = 10
n_trials = 3
epochs = 1
# cifar 10 を読み込む-------------------------------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)

# ラベル
labels = np.array([
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
])

# データの前処理
# ラベルをバイナリクラスにする(yの値を10この数値の配列に変換している)
categorical_y_train = to_categorical(y_train, 10)
categorical_y_test = to_categorical(y_test, 10)

# 正解ラベルの中身の種類 (0~9)をlistに格納
class_list = np.unique(y_train).tolist()
num_class = len(class_list)
print("num_class:", num_class)

# --------------------------------------------------------------------------------------------------------------

# データ拡張----------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------

# CNN の作成----------------------------------------------------------------------------------------------------

inputs = Input(shape=(32, 32, 3))
# flatten_layer = Flatten()(inputs)
x = Conv2D(128, (3,3), padding = "same", activation="relu")(inputs)

# モデル作成時のパラメータを設定
n_layers = 3
units = 128
kernel = 3
pool = 2
drop = 0.25

for i in range(n_layers):
    x = Conv2D(units/2**(i-1), (kernel,kernel), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(pool,pool), strides=(pool,pool))(x)
    x = Dropout(drop)(x)

# 平滑化
x = Flatten()(x)

# 全統合
x = Dense(512, activation="relu")(x)
x = Dropout(0.6)(x)
predictions = Dense(num_class, activation="softmax")(x)

model = Model(inputs=inputs, outputs=predictions)

model.compile(loss="categorical_crossentropy",
                optimizer="adam",
                metrics=["accuracy"]
                )


    

# def objective(trial, X_train, categorical_y_train):
#     # モデルの作成
#     model = create_model(trial)

#     # batch_size範囲を指定
#     batch_size = trial.suggest_int('batch_size', 16, 256, step=16)

#     # KFold のオブジェクトを作成
#     kf = KFold(n_splits=folds, shuffle=True, random_state=42)

#     for train_index, valid_index in kf.split(X_train, categorical_y_train):
#         X_train_subset = X_train[train_index]
#         y_train_subset = categorical_y_train[train_index]
#         X_valid_subset = X_train[valid_index]
#         y_valid_subset = categorical_y_train[valid_index]

#         # 学習
#         model.fit(X_train_subset, y_train_subset,
#                         batch_size=batch_size,
#                         epochs=epochs,
#                         verbose=1,
#                         validation_data=(X_test, y_test),
#                         callbacks=[EarlyStopping(patience=10)]
#         )

#         # 予測
#         y_pred_subset = model.predict(X_valid_subset)
#         # 予測したものを正解と同じ形に変換する
#         y_pred_subset = np.argmax(y_pred_subset, axis=1)

#         # 正解数をカウント
#         num_correct_predictions = np.sum(np.argmax(y_pred_subset, axis=1) 
#                                             == np.squeeze(y_valid_subset))

#         # valid_index の数
#         num_valid_samples = len(y_valid_subset)

#         # 正解率
#         accuracy = num_correct_predictions / num_valid_samples

#         return accuracy
    
# # ベイズ最適化
# sampler = TPESampler()
# study = optuna.create_study(sampler=sampler, direction="maximize")
# study.optimize(lambda trial:objective(trial, X_train, categorical_y_train), n_trials=n_trials)

# model = create_model(study.best_trial)
# --------------------------------------------------------------------------------------------------------------


# 作成したモデルの画像を出力------------------------------------------------------------------------------------
keras.utils.plot_model(model, to_file="/home/gakubu/デスクトップ/ML_git/MLT/ML_11/model.png", 
                        show_shapes=True, show_layer_activations="True")
# --------------------------------------------------------------------------------------------------------------


# 各学習エポックにおける分類の正解率や損失関数の値をプロットして確認する----------------------------------------
# tensorboard を使用

# --------------------------------------------------------------------------------------------------------------


# 誤分類した画像を表示して確認する------------------------------------------------------------------------------
# テストデータに対する予測
y_test_pred = model.predict(X_test)
# 予測したものを元のカテゴリクラスに変換する
y_test_pred = np.argmax(y_test_pred, axis=1)
y_test_pred_labels = y_test_pred.reshape(-1, 1)
# 予測と正解ラベルが異なる画像のインデックスを取得
misclassified_indices = np.where(np.argmax(y_test_pred, axis=1) != np.argmax(categorical_y_test, axis=1))[0]
# 表示する画像の枚数
num_images_to_display = min(50, len(misclassified_indices))
# グリッドの行数と列数を計算
num_rows = int(np.sqrt(num_images_to_display))
num_cols = int(np.ceil(num_images_to_display / num_rows))
# 誤分類された画像を表示
for i in range(num_images_to_display):
    index = misclassified_indices[i]

    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(X_test[index])
    plt.axis("off")
    titles = labels[y_test_pred_labels[index][0]] + "," + labels[y_test[index][0]]
    plt.title(titles)
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------------------------------------


# TSNE を使って畳み込み演算の繰り返しで抽出した特徴量を---------------------------------------------------------
# 低次元化して二次元でプロットする

# --------------------------------------------------------------------------------------------------------------

print("Finished Program")