import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from keras.datasets import cifar10
from keras.utils import to_categorical

# 画像を読み込み
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
img = X_train[0]

# 画像をnumpy配列に変換
img_array = image.img_to_array(img)

# 表示
plt.imshow(img)
plt.axis('off')              # 軸を非表示にする
plt.show()

# 画像表示の準備
def show(datagen, img):

    # 画像をbatch_sizeの数ずつdataに入れる
    for i, data in enumerate(datagen.flow(img,batch_size = 1, seed = 42)):
        show_img = array_to_img(data[0], scale = False)
        # 2 x 3 の画像表示の枠を設定＋枠の指定
        plt.subplot(2, 3, i+1)
        
        plt.imshow(show_img)
        plt.axis('off')
        # 6回目で繰り返しを強制終了
        if i == 5:
            plt.show()
            return
        
# print(img_array.shape)
# 配列に次元を追加
img_win = img_array[np.newaxis, :, :, :]
# # 次元追加後の配列の形
# print(img_win.shape)



# -180度~+180度の間でランダムに回転するImageDataGeneratorを作成
rotation_datagen = ImageDataGenerator(rotation_range = 180)
# 画像を表示
show(rotation_datagen, img_win)


# -------------------------------------なんか上下と左右逆になってる----------------------------------
# 左右平行移動　引数はwidth_shift_range
# (50) ：指定されたピクセル(-50~50)の範囲で左右にランダムに動かす
# ([50, 100]) ：指定されたピクセル(-100, -50, 50, 100) のうち左右にランダムに動かす
# (0.5) ：指定された値 x 画像の横幅(-320~ 320) の範囲で左右にランダムに動かす
width_datagen = ImageDataGenerator(width_shift_range = 0.5)
show(width_datagen, img_win)

# 上下平行移動
height_datagen = ImageDataGenerator(height_shift_range=0.5)
show(height_datagen, img_win)

# ----------------------------------------------------------------------------------------------------

# 拡大、縮小
zoom_datagen = ImageDataGenerator(zoom_range=[0.5, 1.5])
show(zoom_datagen, img_win)

# 画像のせん断
# shear_range：せん断する角度（度）
shear_datagen = ImageDataGenerator(shear_range=30)
show(shear_datagen, img_win)

# 左右反転
horizontal_datagen = ImageDataGenerator(horizontal_flip = True)
show(horizontal_datagen, img_win)

# 上下反転
vertical_datagen = ImageDataGenerator(vertical_flip=True)
show(vertical_datagen, img_win)

# 明るさの調整
brightness_datagen = ImageDataGenerator(brightness_range=[0.3, 0.8])
show(brightness_datagen, img_win)

