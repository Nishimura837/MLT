import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

    #csvファイルをpandasを使って読み込む
    #csvファイルが保存されているルートディレクトリのパス
root_directory = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/"
    #各officeのinputdataをデータフレームとして読み込む
    #フォルダごとに処理を繰り返す
for folder_name in os.listdir(root_directory):
    folder_path = os.path.join(root_directory, folder_name)
    csv_file_path = os.path.join(folder_path, "inputdata.csv")  #各フォルダ内のinputdata.csvファイルのパス
    if os.path.isfile(csv_file_path):
        df_name = f"df_input_{folder_name}"   #データフレーム名をフォルダ名に基づいて作成
        #csvファイルをデータフレームとして読み込む
        globals()[df_name] = pd.read_csv(csv_file_path)
        #カテゴリ変数である"exhaust"を[Label Encoding]により数値化する
        #"exhaust"の値に応じて、"a"なら"0"、"b"なら1、"off"なら"2"に変換する
        globals()[df_name].loc[globals()[df_name]['exhaust'] == "a", 'exhaust'] = 0
        globals()[df_name].loc[globals()[df_name]['exhaust'] == "b", 'exhaust'] = 1
        globals()[df_name].loc[globals()[df_name]['exhaust'] == "off", 'exhaust'] = 2

        #'aircon_position_x', 'aircon_position_y'の列があった場合にその列を削除する
        #今回は片方があればどちらも存在するから片方の条件で両方削除する
        if 'aircon_position_x' in globals()[df_name].columns :
            globals()[df_name] = globals()[df_name].drop(columns=['aircon_position_x', 'aircon_position_y'])


    #作成されたすべてのデータフレームの名前を取得
df_names = [var_name for var_name in globals() if isinstance(globals()[var_name], pd.DataFrame)]
    #countfrom2secpatientAverage.csvをデータフレームとして読み込む
df_count_from2sec = pd.read_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/"\
                                "count_from2sec_patientAverage.csv", header=0)
dfc = df_count_from2sec
    ##ここまでで必要なデータはすべて読み込み済み

    # casename,case_name列をキーにしてinputdataとdfcのRoI列を結合
for name in df_names:
    df_name = f"R_{name}"   #データフレーム名をフォルダ名に基づいて作成
    # 名前を使用してデータフレームにアクセス
    name = globals()[name]  #組み込み関数の globals() を呼び出すと、グローバルスコープに定義されている関数、変数のディクショナリを取得できます
    globals()[df_name] = pd.merge(name, dfc, left_on='case_name', right_on='casename', how='left')
    globals()[df_name] = globals()[df_name].drop(columns=['casename'])
    # print(globals()[df_name])
    print(globals()[df_name].shape)

    #作成されたすべてのデータフレームの名前を取得
df_names = [var_name for var_name in globals() if isinstance(globals()[var_name], pd.DataFrame)]
    # 'R' を含むデータフレームの名前を格納するリストを初期化
R_names = []
for variable_name in df_names:
    if 'R' in variable_name:
        R_names.append(variable_name)

print(R_names)
    # 空のリストを作成してデータフレームを格納
df_list = []
    # データフレームをリストに追加
for df_name in R_names:
    df = globals()[df_name]  # データフレーム名からデータフレームを取得
    df_list.append(df)

X =  pd.concat(df_list, axis=0, ignore_index=True)

    ##トレーニングデータとテストデータに分割する（office10をテストデータ、他はトレーニングデータ）
    #casenameにoffice10を含む行を抽出
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
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 標準化されたデータを新しいデータフレームに格納
X_train = pd.DataFrame(X_train_sc, columns=X_train.columns)
X_test = pd.DataFrame(X_test_sc, columns=X_test.columns)

#標準化されたデータX_train,y_train,X_test,y_testをcsvファイルとして出力
X_train.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_train.csv", encoding='utf_8_sig', index=False)
y_train.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_train.csv", encoding='utf_8_sig', index=False)
X_test.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_test.csv",encoding='utf_8_sig', index=False)
y_test.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_test.csv",encoding='utf_8_sig', index=False)
df_train.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train.csv",encoding='utf_8_sig', index=False)
df_test.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test.csv",encoding='utf_8_sig', index=False)