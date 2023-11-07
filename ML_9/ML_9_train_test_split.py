import os #OSモジュールの読み込み
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#csvファイルの読み込み

def read_inputdata_csv_files(root_folder):
    dataframes = []  # 読み込んだデータフレームを格納するリスト

    for root, dirs, files in os.walk(root_folder): #os.walk()でファイルやディレクトリの一覧取得
        if "inputdata.csv" in files:  # "inputdata.csv"ファイルがある場合
            csv_path = os.path.join(root, "inputdata.csv") 
            df = pd.read_csv(csv_path)
            dataframes.append(df) #listの末尾に要素を追加

    return dataframes

root_folder = "/home/gakubu/デスクトップ/MLTgit/MLT/ML_9" #フォルダのパス指定

dataframes = read_inputdata_csv_files(root_folder)

#前処理
df_list = []
for df in dataframes:
    # ここでデータフレームを操作できる
    category_mapping = {"a": 1, "b": 2, "off": 3}
    df['exhaust'] = df['exhaust'].map(category_mapping) #インデックスの指定
    #指定した列の削除
    #column_to_delete = 'aircon_position_x'

    if 'aircon_position_x' in df.columns:
        df = df.drop(columns = ['aircon_position_x', 'aircon_position_y'])
    df_list.append(df)
    #print(df.head())  # 例: 最初の5行を表示

combined_df = pd.concat(df_list,ignore_index = True)

#combined_df.to_csv('test_combined.csv', index = False) #csvファイルの出力

#以下のファイルを読み込む
file_path = os.path.join(root_folder, 'count_from2sec_patientAverage.csv')

df_cfp = pd.read_csv(file_path)

#コラムを指定して結合
key_column = 'case_name'
merged_df = pd.merge(combined_df, df_cfp, how = 'left', on = key_column)
#指定の列を削除
merged_df = merged_df.drop(columns = ['num_drop', 'volume[ml]'])
merged_df.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_0/merged.csv", index = False) #csvファイルの出力

#トレーニングデータとテストデータの分割
df_train = merged_df[~merged_df['case_name'].str.contains('office10')] #指定したキーワードのつく行を削除
df_train.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_train.csv", index = False)

df_test = merged_df[merged_df['case_name'].str.contains('office10')] #指定したキーワードのつく行だけを残す
df_test.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_test.csv", index = False)

#説明変数と目的変数の分割
X_train = df_train.drop(columns = ['case_name', 'RoI'])
y_train = df_train['RoI']
X_test = df_test.drop(columns = ['case_name', 'RoI'])
y_test = df_test['RoI']

#説明変数の標準化
scaler = StandardScaler()
scaler.fit(X_train) #テストデータの変換には、 訓練データで計算された統計データを使用する必要がある
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

df_X_train_sc = pd.DataFrame(X_train_sc, columns = X_train.columns)
df_X_test_sc = pd.DataFrame(X_test_sc, columns = X_test.columns)

df_X_train_sc.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_train_sc.csv", index = False)
y_train.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_train.csv", index = False)
df_X_test_sc.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/X_test_sc.csv", index = False)
y_test.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/y_test.csv", index = False)

