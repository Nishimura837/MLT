import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing import image

# 仮想的な画像データとラベルデータを生成（実際のデータに置き換えてください）
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
categorical_y_train = to_categorical(y_train, 10)

# X_train = np.random.rand(100, 150, 150, 3)  # 100枚の仮想的なRGB画像
# categorical_y_train = np.random.randint(2, size=(100,))  # 2クラスの仮想的なラベルデータ（0または1）
for i in range(5):
    plt.imshow(X_train[i])
    plt.show()


# ImageDataGeneratorの設定
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# データジェネレータの作成
data_generator = datagen.flow(X_train, categorical_y_train, batch_size=32)

# 1つのバッチデータを取得
batch_data, batch_labels = data_generator.next()
print(batch_data)

# バッチ内の1つ目の画像を表示
plt.imshow(batch_data[0])
plt.show()

