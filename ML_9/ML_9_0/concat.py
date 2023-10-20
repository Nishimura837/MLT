import pandas as pd

# サンプルのデータフレーム df1, df2, df3 を作成
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})
df3 = pd.DataFrame({'A': [13, 14, 15], 'B': [16, 17, 18]})

# データフレームを縦に結合
result = pd.concat([df1, df2, df3], ignore_index=True)

# 結果を表示
print(result)