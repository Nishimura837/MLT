import numpy as np

data = np.array([[0.3, 1, 0.7],
                 [5, 4, 7],
                 [12, 3, 9]])

# 各行における最大値のインデックスを取得
max_indices = np.argmax(data, axis=1)

print("各行における最大値の場所（列のインデックス）:")
print(max_indices)