import os
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

#csvファイルをpandasを使って読み込む
#csvファイルが保存されているルートディレクトリのパス
root_directory = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/"
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


        # print(df_name)
        # print(globals()[df_name])

#作成されたすべてのデータフレームの名前を取得
df_names = [var_name for var_name in globals() if isinstance(globals()[var_name], pd.DataFrame)]
# #データフレームの名前を表示
# for name in df_names:
#     print(name)


#countfrom2secpatientAverage.csvをデータフレームとして読み込む
df_count_from2sec = pd.read_csv("/home/gakubu/デスクトップ/python_git/my_project/ML_9/"\
                                "count_from2sec_patientAverage.csv", header=0)
dfc = df_count_from2sec


            ##osライブラリについて
            # print(os.getcwd())  #現在の作業ディレクトリを返す
            # print(os.listdir(root_directory))   #パス上のファイルやディレクトリの一覧をリストで返す
            # print(os.listdir('.'))  #現在の作業ディレクトリ内にあるファイル一覧を返す
            # os.mkdir('adir')    #'adir'というディレクトリの作成
            # os.rmdir('adir')    #'adir'というディレクトリの削除
            # print(os.path.split(os.getcwd()))   #[親ディレクトリのパス, 作業ディレクトリ] という 2 要素のリストを返す
            # print(os.path.join(root_directory, 'ML_9_0.py'))    #パスとファイルをつなげて返す



#データをオフィス毎に色分けしてプロットする
#（横軸）RoI、（縦軸）office毎にインデックスをつけたその値 
#office毎に分けるため、DFに新たにoffice_nameの列を追加する


#dfcに'office_name'列を追加
for folder_name in os.listdir(root_directory):
    for index,row in dfc.iterrows() :
        if folder_name in row['casename']:                  #casenameに'folder_nameが含まれているかどうか
            dfc.at[index, 'office_name'] = folder_name

# print(dfc)

# 各オフィス名に対する色を 'tab20' カラーマップから取得
office_names = dfc['office_name'].unique()      #unique()メソッドは指定した列内の一意の値の配列を返す（重複を取り除く）
colors = plt.cm.tab20(range(len(office_names))) #tab20から配列office_namesの長さ分の色の配列colorsを返す

# オフィス名と色の対応を辞書に格納
# zip関数は２つ以上のリストを取り、それらの対応する要素をペアにしてイテレータを返す。
#この場合、office_namesとcolorsの２つのリストをペアにし、対応する要素同士を取得する。
# =以降はofficeをキーとしてそれに対応するcolorが"値"として格納される辞書を作成
office_color_mapping = {office: color for office, color in zip(office_names, colors)}

# 'office_name' 列を数値（色情報に対応する数値）に変換
# 'office_num'　を追加
dfc['office_num'] = dfc['office_name'].map(office_color_mapping)

dfc.plot.scatter(x='RoI', y='office_name', c=dfc['office_num'])
plt.title('RoI for each office')
#plt.show()
plt.savefig("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_0/RoI for each office.pdf", format='pdf')       


# casename,case_name列をキーにしてinputdataとdfcのRoI列を結合
for name in df_names:
    df_name = f"dfR_{name}"   #データフレーム名をフォルダ名に基づいて作成
    # 名前を使用してデータフレームにアクセス
    name = globals()[name]  #組み込み関数の globals() を呼び出すと、グローバルスコープに定義されている関数、変数のディクショナリを取得できます
    globals()[df_name] = pd.merge(name, dfc, left_on='case_name', right_on='casename', how='left')
    globals()[df_name] = globals()[df_name].drop(columns=['casename'])
    # print(globals()[df_name])
    # print(globals()[df_name].shape)

#作成されたすべてのデータフレームの名前を取得
df_names = [var_name for var_name in globals() if isinstance(globals()[var_name], pd.DataFrame)]


# 'dfR' を含むデータフレームの名前を格納するリストを初期化
dfR_names = []
for variable_name in df_names:
    if 'dfR' in variable_name:
        dfR_names.append(variable_name)

# print(dfR_names)

# ##説明変数と目的変数の相関関係を探ってヒートマップを作成する
# for office in dfR_names :
#     df_office = globals()[office]
#     X = df_office.drop(columns=['case_name','office_name','office_num'])
#     print(X)
#     # 相関行列を計算
#     correlation_matrix = X.corr()
#     # ヒートマップを作成
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#     plt.title(f"Correlation Heatmap_{office}")
#     #plt.show()
#     # ファイル名を組み立てて保存
#     file_name = f"Correlation_Heatmap_{office}.pdf"
#     save_path = "/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_0/" + file_name   
#     plt.savefig(save_path, format='pdf')
#     plt.close()

#すべてのofficeデータを１つのデータフレームにまとめる
# データフレームを縦に結合

# 空のリストを作成してデータフレームを格納
df_list = []

# データフレームをリストに追加
for df_name in dfR_names:
    df = globals()[df_name]  # データフレーム名からデータフレームを取得
    df_list.append(df)
# データフレームの結合
df_concat =  pd.concat(df_list, axis=0, ignore_index=True)

X = df_concat.drop(columns=['case_name','office_name','office_num'])
# print(X)
# 相関行列を計算
correlation_matrix = X.corr()
# ヒートマップを作成
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='seismic', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
#plt.show()  
plt.savefig("/home/gakubu/デスクトップ/python_git/my_project/ML_9/ML_9_0/Correlation Heatmap.pdf", format='pdf') 
plt.close()


##トレーニングデータとテストデータに分割する（office10をテストデータ、他はトレーニングデータ）
#casenameにoffice10を含む行を抽出
condition1 = df_concat['case_name'].str.contains('office10')
df_test = df_concat[condition1]
condition2 = ~df_concat['case_name'].str.contains('office10')
df_train = df_concat[condition2]
# print(df_test)
# print(df_train)