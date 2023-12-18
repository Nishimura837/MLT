import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# ダミーデータの生成（実際のデータを使用してください）
(X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# Functional APIを使用してモデルを構築
inputs = layers.Input(shape=(32, 32, 3))
conv1 = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
maxpool1 = layers.MaxPooling2D((2, 2))(conv1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu')(maxpool1)
maxpool2 = layers.MaxPooling2D((2, 2))(conv2)
conv3 = layers.Conv2D(64, (3, 3), activation='relu')(maxpool2)
flatten = layers.Flatten()(conv3)
dense1 = layers.Dense(64, activation='relu')(flatten)
outputs = layers.Dense(10, activation='softmax')(dense1)

# モデルを作成
model = models.Model(inputs=inputs, outputs=outputs)

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの訓練
model.fit(X_train, y_train, epochs=1, batch_size=64)

# Functional APIを使用して中間層の出力を取得
intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)  # Flatten レイヤーの手前の層
intermediate_output = intermediate_layer_model.predict(X_train)
print("intermediate_output_shape:", intermediate_output.shape)
# NaNやinfを含む行を削除
cleaned_output = intermediate_output[~np.any(np.isnan(intermediate_output) | np.isinf(intermediate_output), axis=1)]

# t-SNEを用いて特徴量を2次元に変換
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(cleaned_output)

# 各クラスごとに色を設定
num_classes = len(np.unique(y_train))
colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

# プロット
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    indices = np.where(y_train.flatten() == i)[0]
    indices_cleaned = np.intersect1d(indices, np.where(~np.any(np.isnan(intermediate_output) | np.isinf(intermediate_output), axis=1)))
    plt.scatter(tsne_result[indices_cleaned, 0], tsne_result[indices_cleaned, 1], color=colors[i], label=f'Class {i}', alpha=0.7)

plt.title('t-SNE Visualization of Intermediate Features')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
