import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# カラーマップの色を定義
colors = [  (1, 0, 0),   # 赤
            (0, 1, 0),   # 緑
            (0, 0, 1)]   # 青

# カラーマップを作成
cmap = ListedColormap(colors)

# データ生成
x = [0, 1, 2]
y = [0, 1, 2]
z_values = [0, 1, 2]  # 引数Zの値

# プロット
plt.scatter(x, y, c=z_values, cmap=cmap, s=100)
plt.colorbar(ticks=[0, 1, 2], label='Z')
plt.show()


#-------------------------
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_custom_colormap(value):
    if value == 0:
        color = 'red'
    elif value == 1:
        color = 'green'
    elif value == 2:
        color = 'blue'
    else:
        raise ValueError("Invalid value. Only 0, 1, and 2 are allowed.")

    cmap = LinearSegmentedColormap.from_list('custom_cmap', [color, color], N=256)
    return cmap

# テスト用の値を指定してカラーマップを作成
value = 1
custom_cmap = create_custom_colormap(value)

# カラーバーを表示してカラーマップを表示
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap), cax=ax, orientation='horizontal')
cbar.set_label('Color Map')

plt.show()
