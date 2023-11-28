import tensorflow as tf
from tensorflow import keras 
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))      # 404 examples, 13 features
print("Testing set: {}".format(test_data.shape))        # 102 examples, 13 features

print(train_data[0])    # Display sample features, notice the differenr scales


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT']


# 正規化する
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean)/ std
test_data = (test_data - mean)/ std


# モデルを作成する
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,input_shape=(train_data.shape[1],)),
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
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mae']),
                label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mae']),
                label='Val loss')
    plt.legend()
    plt.ylim([0,5])

plot_history(history)

model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])
plot_history(history)
plt.show()