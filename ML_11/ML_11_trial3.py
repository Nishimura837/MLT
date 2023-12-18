# ML_11_trial2.py に加えて、モデル作成時にoptunaを使って
# パラメータの最適化を行う

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
import tensorflow as tf
import datetime
import optuna
from sklearn.model_selection import KFold
from optuna.samplers import TPESampler, GridSampler
from sklearn.manifold import TSNE



# cifar10を読み込む
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

# 画像表示のための関数
def image_show(x, y, labels):

    for i in range(50):
        plt.subplot(5, 10, i+1)
        # 軸を表示しない
        plt.axis("off")
        plt.title(labels[y[i][0]])
        plt.imshow(x[i])
        
    plt.tight_layout()
    plt.show()
    return

# トレーニングデータの画像とラベルをセットで表示してみる
image_show(X_train, y_train, labels)

# 正解ラベルの中身の種類 (0~9)をlistに格納
class_list = np.unique(y_train).tolist()
num_class = len(class_list)
print(class_list)

# データの前処理
# ラベルをバイナリクラスにする(yの値を10この数値の配列に変換している)
categorical_y_train = to_categorical(y_train, 10)
categorical_y_test = to_categorical(y_test, 10)

print(categorical_y_train[0])


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


#モデルを構築
folds = 10

def create_model(trial):
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(128, (3,3), padding = "same", activation="relu")(inputs)

    # モデル作成時のパラメータを設定
    n_layers = trial.suggest_int('n_layers', 1, 5)
    units = trial.suggest_int('n_units', 8, 256, step=8)
    kernel = trial.suggest_int('kernel', 3, 8)
    pool = trial.suggest_int('pool', 2, 5)
    drop = trial.suggest_float('drop', 0.1, 0.75, step=0.05)

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
    

    return model

def objective(trial):
    # モデルの作成
    model = create_model(trial)

    # batch_size範囲を指定
    batch_size = trial.suggest_int('batch_size', 16, 256, step=16)

    # KFold のオブジェクトを作成
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    
    for train_index, valid_index in kf.split(augmentation_train_generator):
        X_train_subset = X_train[train_index]
        y_train_subset = categorical_y_train[train_index]
        X_valid_subset = X_train[valid_index]
        y_valid_subset = categorical_y_train[valid_index]

        # 学習
        model.fit(X_train_subset, y_train_subset,
                        batch_size=batch_size,
                        epochs=5,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(patience=10)]
        )

        # 予測
        y_pred_subset = model.predict(X_valid_subset)

        # 正解数をカウント
        num_correct_predictions = np.sum(np.argmax(y_pred_subset, axis=1) 
                                            == np.squeeze(y_valid_subset))

        # valid_index の数
        num_valid_samples = len(y_valid_subset)

        # 正解率
        accuracy = num_correct_predictions / num_valid_samples

        return accuracy


# ベイズ最適化
sampler = TPESampler()
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=50)

augmentation_model = create_model(study.best_trial)

# モデルの画像を作成して表示
keras.utils.plot_model(augmentation_model, "model.png")

# tensorboard を使って各学習エポックにおける正解率や損失関数の値をプロットする
log_dir = "/home/gakubu/デスクトップ/ML_git/MLT" + datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# データ拡張したものを使って学習
augmentation_result = augmentation_model.fit(
    augmentation_train_generator, 
    steps_per_epoch = len(X_train)/32, 
    epochs = 20,
    callbacks=[tensorboard_callback]
)


# テストデータに対する予測
y_test_pred = augmentation_model.predict(X_test)

# 予測と正解ラベルが異なる画像のインデックスを取得
misclassified_indices = np.where(np.argmax(y_test_pred, axis=1) != np.squeeze(categorical_y_test))[0]

# 表示する画像の枚数
num_images_to_display = min(25, len(misclassified_indices))

# グリッドの行数と列数を計算
num_rows = int(np.squrt(num_images_to_display))
num_cols = int(np.ceil(num_images_to_display / num_rows))

# 誤分類された画像を表示
for i in range(num_images_to_display):
    index = misclassified_indices[i]
    predicted_class = np.argmax(y_test_pred[index])
    true_class = y_test[index, 0]

    plt.subplot(num_rows, num_cols, i+1)
    plt.imshow(X_test[index])
    plt.axis("off")

plt.tight_layout()
plt.show()


# 特徴量の可視化
# モデルの最後の層を取り除いて、特徴量を取得する
feature_extractor_model = Model(inputs=augmentation_model.input, outputs=augmentation_model.layers[-2].output)

# 予測したい画像を入力して、最後の分類層の前の出力を取得
features = feature_extractor_model.predict(X_test)

# t-SNE を適用
tsne = TSNE(n_components=2)
embedded_features = tsne.fit_transform(features)

# 可視化
plt.scatter(embedded_features[:,0],embedded_features[:,1])
plt.title("t-SNE Visualization of CNN Features")
plt.show()

print("Finished ML_11_trial3.py")