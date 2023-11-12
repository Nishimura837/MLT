import os
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

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
df_merged = df_merged.drop(columns = ["num_drop", "volume[ml]"])

#データをオフィス毎に色分けしてプロットする
#（横軸）RoI、（縦軸）office毎にインデックスをつけたその値 
#office毎に分けるため、DFに新たにoffice_nameの列を追加する
#dfcに'office_name'列を追加
for folder_name in os.listdir(root_directory):
    for index,row in df_merged.iterrows() :
        if folder_name + '_' in row['case_name']:                  #casenameに'folder_nameが含まれているかどうか
            df_merged.at[index, 'office_name'] = folder_name  

df_merged.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_0/dfc.csv",encoding='utf_8_sig', index=False)
# 各オフィス名に対する色を 'tab20' カラーマップから取得
office_names = df_merged['office_name'].unique()      #unique()メソッドは指定した列内の一意の値の配列を返す（重複を取り除く）
colors = plt.cm.tab20(range(len(office_names))) #tab20から配列office_namesの長さ分の色の配列colorsを返す

# オフィス名と色の対応を辞書に格納
# zip関数は２つ以上のリストを取り、それらの対応する要素をペアにしてイテレータを返す。
#この場合、office_namesとcolorsの２つのリストをペアにし、対応する要素同士を取得する。
# =以降はofficeをキーとしてそれに対応するcolorが"値"として格納される辞書を作成
office_color_mapping = {office: color for office, color in zip(office_names, colors)}

# 'office_name' 列を数値（色情報に対応する数値）に変換
# 'office_num'　を追加
df_merged['office_num'] = df_merged['office_name'].map(office_color_mapping)

df_merged.plot.scatter(x='RoI', y='office_name', c=df_merged['office_num'])
plt.title('RoI for each office')
#plt.show()
plt.savefig("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_0/RoI for each office modified.pdf", format='pdf')       


X = df_merged.drop(columns=['case_name','office_name','office_num'])
# print(X)
# 相関行列を計算
correlation_matrix = X.corr()
# ヒートマップを作成
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='seismic', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap modified")
#plt.show()  
plt.savefig("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_0/Correlation Heatmap modified.pdf", format='pdf') 
plt.close()


##トレーニングデータとテストデータに分割する（office10をテストデータ、他はトレーニングデータ）
#case_nameにoffice10を含む行を抽出
condition1 = df_merged['case_name'].str.contains('office10')
df_test = df_merged[condition1]
condition2 = ~df_merged['case_name'].str.contains('office10')
df_train = df_merged[condition2]
df_test.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test_modified.csv",encoding='utf_8_sig', index=False)
df_train.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train_modified.csv",encoding='utf_8_sig', index=False)