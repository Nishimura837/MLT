import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.preprocessing .image import ImageDataGenerator, array_to_img, img_to_array
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping



# cifar10を読み込む
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# NUM 番目の画像を表示
NUM = 1
plt.imshow(X_train[NUM])



# ラベル
label_dict = {
    0: "飛行機",
    1: "自転車",
    2: "鳥",
    3: "猫",
    4: "鹿",
    5: "犬",
    6: "カエル",
    7: "馬",
    8: "船",
    9: "トラック",
}

# 正解ラベルの中身の種類 (0~9)をlistに格納
class_list = np.unique(y_train).tolist()
num_class = len(class_list)
print(class_list)

# 正規化する
X_train = X_train.astype("float32")/255.0
X_test = X_test.astype("float32")/255.0
# yを２値配列に変換
y_train = to_categorical(y_train, num_class)
y_test = to_categorical(y_test, num_class)


# data augmentation の実施










# 画像表示用の関数を定義
# def show_image(datagen, img):
#     # 表示サイズを指定
#     plt.figure(figuresize=(10, 5))

#     # 画像をbatch_sizeの数ずつdataに入れる
#     for i, data in enumerate(datagen.flow(img, batch_size=1024, seed=42)):
#         # 表示のためにnumpy配列からimgに変換する
#         show_img = array_to_img(data[0], scale=False)
#         # 2**5 x 2**5 の画像表示の枠を設定＋枠の指定
#         plt.subplot(2**5, 2**5, i+1)
#         # 軸を表示しない
#         plt.xticks(color="None")
#         plt.yticks(color="None")
#         plt.tick_params(bottom = False, left = False)
#         # 画像を表示
#         plt.imshow(show_img)

#         return


# 学習
def train(X_train, y_train, X_test, y_test):
    inputs = Input(shape = (32, 32, 3))

    x = Conv2D(128, (3,3), padding="same", activation="relu")(inputs)
    x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    # 平滑化
    x = Flatten()(x)
    
    # 全結合
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.6)(x)
    predictions = Dense(num_class, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=predictions)

    # 学習の設定
    model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=["accuracy"])

    # 学習
    history = model.fit(X_train, y_train,
                        batch_size=1024,
                        epochs=5,
                        verbose=1,
                        validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(patience=10)]
                        )
    
    # モデルの構造と重みを保存
    model.save("./cnn1.h5")

    return model, history


# テストデータで正解率を算出
def test_accuracy(model, X_test, y_test):
    # 全正解数
    sum_correct = 0

    # クラスごとの正解率
    class_total = [0 for i in range(num_class)]
    class_correct = [0. for i in range(num_class)]

    for i, data in enumerate(X_test):
        pred = model.predict(np.array([data])) # np.array([np.array])で(32,32,3)→(1,32,32,3)に整形
        pred = pred.reshape(pred.shape[1])  # predを2次元配列で出てくるので、1次元配列に変換
        pred_index = np.argmax(pred)  # 一番確率が高い引数を取得
        label = np.argmax(y_test[i]) # yは二値配列にしているので、np.argmaxで中身を取り出す(0～9)
        sum_correct += (1 if pred_index==label else 0) # y_testと一致した個数を累積
        class_total[label] += 1 # label番目の個数を+1
        class_correct[label] += (1 if pred_index==label else 0) # 正解ならlabel番目の正解数を+1

    print("-"*100)
    print("正解数：", sum_correct)
    print("データ数：", len(X_test))
    print("正解率：", (sum_correct/len(X_test)*100))

    print("-"*100)
    for i in range(num_class):
        print("%5s クラスの正解率：%.1f %%" %(label_dict[i], class_correct[i]/class_total[i]*100))

# 学習過程を可視化
def plot_fig(history):
    print("-"*100)
    print("BatchNormalizationなし")
    # 描画する領域を設定
    plt.figure(1, figsize=(13,4))
    plt.subplots_adjust(wspace=0.5)

    # 学習曲線
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.title("train and valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()

    # 精度表示
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.title("train and valid accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()

    plt.show()


# 実行
# メイン関数
def main():
    model, history = train(X_train, y_train, X_test, y_test)
    test_accuracy(model, X_test, y_test)
    plot_fig(history)

main()

