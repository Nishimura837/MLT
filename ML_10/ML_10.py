import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import RMSprop



#test.csv,train.csvを取得
df_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train.csv"
df_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test.csv"
df_train = pd.read_csv(df_train_path)
df_train = df_train.drop(columns="case_name")
df_test = pd.read_csv(df_test_path)
df_test = df_test.drop(columns="case_name")
nd_train = df_train.to_numpy()
nd_train = nd_train.astype(float)
nd_test = df_test.to_numpy()
nd_test = nd_test.astype(float)
columns = np.array(df_train)

print(nd_train)

# モデルを作成する
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,input_shape=(nd_train.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = RMSprop(0.001)

    model.compile( loss="mse",
                optimizer=optimizer,
                metrics=["mae"])
    
    return model

model = build_model()
model.summary()

# モデルを訓練する
# モデルは500エポックの間訓練されて、訓練と検証精度をhistoryオブジェクトに記録する
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 500
history = model.fit(nd_train, columns, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
print(history.history)

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mae']),
                label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mae']),
                label='Val loss')
    plt.legend()
    

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

plot_history(history)
plt.show()

# 
