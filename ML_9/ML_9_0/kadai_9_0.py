import os #OSモジュールの読み込み
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # カラーマップのインポート
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

#ヒートマップの作成
start_col = 'aircon'

# 指定した行と列の範囲を選択して相関を計算
subset = merged_df.iloc[:, merged_df.columns.get_loc(start_col):]
correlation_matrix = subset.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, square=True, cbar=True, annot=None, cmap='Blues')
plt.show()

def get_office_name(case_name):
    name_parts = case_name.split('_',2)  # _で分割
    if len(name_parts) > 0:
        return name_parts[0]
    else:
        return case_name  # 分割できない場合は元の行名を返す

#データをオフィスごとにプロット
merged_df['office_name'] = merged_df['case_name'].map(get_office_name)
#plt.legend()
#plt.show()
merged_df.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/ML_9_0/merged_plus.csv", index = False) #csvファイルの出力

def set_color(i):
    if i == 'office1':
        return "b"  # blue
    elif i == 'office2':
        return "g"  # green
    elif i == 'office3':
        return "r"  # red
    elif i == 'office4':
        return "c" 
    elif i == 'office5':
        return "m" 
    elif i == 'office6':
        return "y" 
    elif i == 'office7':
        return "k" 
    elif i == 'office8':
        return "tomato"  
    elif i == 'office9':
        return "gold" 
    elif i == 'office10':
        return "silver" 
    elif i == 'office14':
        return "violet" 
    elif i == 'office15':
        return "crimson"  
    elif i == 'office16':
        return "lightgreen"  
    elif i == 'office22':
        return "orange" 
    elif i == 'office28':
        return "cornsilk" 
      
color_list = list(map(set_color, merged_df['office_name']))
plt.scatter(merged_df['RoI'], merged_df['office_name'], c = color_list)

plt.show() 

#トレーニングデータとテストデータの分割
df_train = merged_df[~merged_df['case_name'].str.contains('office10')] #指定したキーワードのつく行を削除
df_train = df_train.drop(columns = ['office_name'])
df_train.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_train.csv", index = False)

df_test = merged_df[merged_df['case_name'].str.contains('office10')] #指定したキーワードのつく行だけを残す
df_test.to_csv("/home/gakubu/デスクトップ/MLTgit/MLT/ML_9/df_test.csv", index = False)
