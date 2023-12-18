# モデルの作成、
# 作成したモデルの画像出力
# 各学習エポックでの正解率、損失関数のプロット
# 誤分類した画像の枚数を確認、誤分類した画像を表示
# まで完了

# 特徴量の可視化をする
# optuna による最適化の実行

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.preprocessing .image import ImageDataGenerator
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import datetime
import optuna
from sklearn.model_selection import KFold
from optuna.samplers import TPESampler, GridSampler
from sklearn.manifold import TSNE
from keras.callbacks import TensorBoard


# 最適化の分割数と試行回数、エポック数
folds = 10
n_trials = 3


# cifar 10 を読み込む-------------------------------------------------------------------------------------------
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)

#ラベルの設定
labels = np.array([
    'airplane',  #飛行機
    'automobile',#バイク
    'bird',      #鳥
    'cat',       #猫
    'deer',      #鹿
    'dog',       #犬
    'frog',      #カエル
    'horse',     #馬
    'ship',      #船
    'truck'      #トラック
    ])

#画像の表示のための関数
def image_show(x, y, labels):
    for i in range(30):
        plt.subplot(5, 6, i+1)
        #軸を表示しない
        plt.axis("off")
        #タイトルをラベルの名前で表示
        plt.title(labels[y[i][0]])
        #表示
        plt.imshow(x[i])
    plt.tight_layout()
    plt.show()
    return    

# トレーニングデータの画像とラベルをセットで表示してみる
image_show(X_train, y_train, labels)
image_show(X_test, y_test, labels)

# データの前処理
# ラベルをバイナリクラスにする(yの値を10この数値の配列に変換している)
categorical_y_train = to_categorical(y_train, 10)
categorical_y_test = to_categorical(y_test, 10)

# 正解ラベルの中身の種類 (0~9)をlistに格納
class_list = np.unique(y_train).tolist()
num_class = len(class_list)
print("num_class:", num_class)
# --------------------------------------------------------------------------------------------------------------

# データ拡張(このブロックをコメントアウトするかしないかでデータ拡張の ON, OFF を切り替える)---------------------
# ImageDataGeneratorクラスの作成
augmentation_train_datagen = ImageDataGenerator(
    rotation_range=10,          # 回転
    horizontal_flip=True,       # 左右反転
    height_shift_range=0.2,     # 上下平行移動
    width_shift_range=0.2,      # 左右平行移動
    zoom_range=0.2,             # ランダムにズーム
    channel_shift_range=0.2,    # チャンネルシフト
    rescale=1./255              # スケーリング
)

# バッチの作成
augmentation_train_generator = augmentation_train_datagen.flow(X_train, categorical_y_train, batch_size=32, seed=42)
augmented_X_train, augmented_categorical_y_train = augmentation_train_generator.next()
# データ拡張したものをX_train, categorical_y_trainにする
X_train = augmented_X_train
categorical_y_train = augmented_categorical_y_train
# --------------------------------------------------------------------------------------------------------------

# CNN の作成----------------------------------------------------------------------------------------------------
def create_model(trial):
    # モデル作成時のパラメータを設定
    # モデル作成時のパラメータを設定
    n_layers = trial.suggest_int('n_layers', 1, 5)
    units = trial.suggest_int('n_units', 8, 256, step=8)
    kernel = 3
    pool = 2
    drop = trial.suggest_float('drop', 0.1, 0.75, step=0.05)

    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(64, (3,3), padding = "same", activation="relu")(inputs)

    for i in range(n_layers):
        x = Conv2D(units/2**(i-1), (kernel,kernel), padding="same", activation="relu")(x)
        x = MaxPooling2D(pool_size=(pool,pool))(x)
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
    
    return model
# --------------------------------------------------------------------------------------------------------------

# optuna による最適化
def objective(trial):
    # モデルの作成
    model = create_model(trial)

    # batch_size範囲を指定
    batch_size = trial.suggest_int('batch_size', 16, 256, step=16)
    epochs = trial.suggest_int('epochs', 2, 100, step=7)

    # KFold のオブジェクトを作成
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    
    for train_index, valid_index in kf.split(X_train, categorical_y_train):
        X_train_subset = X_train[train_index]
        y_train_subset = categorical_y_train[train_index]
        X_valid_subset = X_train[valid_index]
        y_valid_subset = categorical_y_train[valid_index]

        # 学習
        model.fit(X_train_subset, y_train_subset,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(X_valid_subset, y_valid_subset),
                        callbacks=[EarlyStopping(patience=10)]
        )

        # 予測
        y_pred_subset = model.predict(X_valid_subset)

        # 正解数をカウント
        num_correct_predictions = np.sum(y_pred_subset == y_valid_subset)

        # valid_index の数
        num_valid_samples = len(y_valid_subset)

        # 正解率
        accuracy = num_correct_predictions / num_valid_samples

        return accuracy


# ベイズ最適化
sampler = TPESampler()
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=n_trials)
# 最適なパラメータの表示
for key, value in study.best_trial.params.items():
    print(f'    {key}: {value}')
# --------------------------------------------------------------------------------------------------------------

# モデルの作成--------------------------------------------------------------------------------------------------
model = create_model(study.best_trial)
# --------------------------------------------------------------------------------------------------------------

# 作成したモデルの画像を出力------------------------------------------------------------------------------------
keras.utils.plot_model(model, to_file="/home/gakubu/デスクトップ/ML_git/MLT/ML_11/model.png", 
                        show_shapes=True, show_layer_activations="True")
# --------------------------------------------------------------------------------------------------------------

# 各学習エポックにおける分類の正解率や損失関数の値をプロットして確認する----------------------------------------
# tensorboard を使用
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# --------------------------------------------------------------------------------------------------------------

# 学習----------------------------------------------------------------------------------------------------------
batch_size = study.best_trial.params['batch_size']
epochs = study.best_trial.params['epochs']
model.fit(X_train, categorical_y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, categorical_y_test),
        callbacks=[tensorboard_callback, EarlyStopping(patience=10)]
)
# --------------------------------------------------------------------------------------------------------------

# 誤分類した画像を表示して確認する------------------------------------------------------------------------------
# テストデータに対する予測
y_test_pred = model.predict(X_test)
# 予測したものを元のカテゴリクラスに変換する
y_test_pred = np.argmax(y_test_pred, axis=1)
y_test_pred_labels = y_test_pred.reshape(-1, 1)
# データの形を確認
print("y_test_pred_labels:", y_test_pred_labels)
print("y_test:", y_test)
# 予測と正解ラベルが異なる画像のインデックスを取得
misclassified_indices = np.where(y_test_pred != np.argmax(categorical_y_test, axis=1))[0]
print("misclassified_indices:", misclassified_indices)
# 表示する画像の枚数
num_images_to_display = min(50, len(misclassified_indices))
print("Num miss categorized:", num_images_to_display)
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

# 正解数をカウント
num_correct = np.sum(y_test_pred == y_test)

# valid_index の数
num_valid_samples = len(y_test)

# 正解率
print("num_correct:", num_correct)
print("num_valid_samples:", num_valid_samples)
accuracy = num_correct / num_valid_samples
print("accuracy:", accuracy)
# --------------------------------------------------------------------------------------------------------------


# TSNE を使って畳み込み演算の繰り返しで抽出した特徴量を---------------------------------------------------------
# 低次元化して二次元でプロットする
# 中間層の出力を取得
# intermediate_output = intermediate_layer_model.predict(X_train)
# # t-SNEによる次元削減
# tsne = TSNE(n_components=2, random_state=42)
# tsne_result = tsne.fit_transform(intermediate_output)
# print(tsne_result.shape)
# # プロット
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_train.flatten(), cmap="tab10")
# plt.legend(*scatter.legend_elements())
# plt.title('t-SNE Visualization of Convolutional Features')
# plt.show()
# # --------------------------------------------------------------------------------------------------------------

print("Finished Program")