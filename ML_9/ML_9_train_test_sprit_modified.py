import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

#csvファイルをpandasを使って読み込む
#csvファイルが保存されているルートディレクトリのパス
root_directory = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/"
#各officeのinputdataをデータフレームとして読み込む
#フォルダごとに処理を繰り返す
df_names = []
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    csv_file_path = os.path.join(folder_path, "inputdata.csv")  #各フォルダ内のinputdata.csvファイルのパス
    if os.path.isfile(csv_file_path):
        df_name = f"df_input_{folder_name}"   #データフレーム名をフォルダ名に基づいて作成

        #csvファイルをデータフレームとして読み込む
        df_name = pd.read_csv(csv_file_path)

        #カテゴリ変数である"exhaust"を[Label Encoding]によりmap関数を用いて数値化する
        #"exhaust"の値に応じて、"a"なら"0"、"b"なら1、"off"なら"2"に変換する
        df_name['exhaust'] = df_name['exhaust'].map({'a': 1, 'b': 2, 'off': 3})

        #'aircon_position_x', 'aircon_position_y'の列があった場合にその列を削除する
        #今回は片方があればどちらも存在するから片方の条件で両方削除する
        if 'aircon_position_x' in df_name.columns :
            df_name = df_name.drop(columns=['aircon_position_x', 'aircon_position_y'])

        df_names.append(df_name)
        print(df_name)

df_input = pd.concat(df_names, axis = 0, ignore_index = True)

#countfrom2secpatientAverage.csvをデータフレームとして読み込む
df_count_from2sec = pd.read_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/"\
                                "count_from2sec_patientAverage.csv", header=0)
dfc = df_count_from2sec.rename(columns={'casename': 'case_name'})

# df_inputとdfcをcase_nameをキーにしてmerge
df_merged = pd.merge(df_input, dfc, how = "inner", on = "case_name")
X = df_merged.drop(columns = ["num_drop", "volume[ml]"])

    ##トレーニングデータとテストデータに分割する（office10をテストデータ、他はトレーニングデータ）
    #case_nameにoffice10を含む行を抽出
condition1 = X['case_name'].str.contains('office10')
df_test = X[condition1]
condition2 = ~X['case_name'].str.contains('office10')
df_train = X[condition2]
print(df_test)
print(df_train)

X_train = df_train.drop(columns=['case_name', 'RoI'])
y_train = df_train["RoI"]
X_test = df_test.drop(columns=['case_name', 'RoI'])
y_test = df_test["RoI"]

#説明変数が複数存在するため、説明変数を標準化する
scaler = StandardScaler()
scaler.fit(X_train)
X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)

# 標準化されたデータを新しいデータフレームに格納
df_X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)
df_X_test = pd.DataFrame(X_test_sc, columns=X_test.columns)

#標準化されたデータX_train,y_train,X_test,y_testをcsvファイルとして出力
df_X_train.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_train.csv", encoding='utf_8_sig', index=False)
y_train.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_train.csv", encoding='utf_8_sig', index=False)
df_X_test.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_test.csv",encoding='utf_8_sig', index=False)
y_test.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_test.csv",encoding='utf_8_sig', index=False)
df_train.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train.csv",encoding='utf_8_sig', index=False)
df_test.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test.csv",encoding='utf_8_sig', index=False)

