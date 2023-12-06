import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np

# cifar10を読み込む
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

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

print(X_test)
print(y_test)