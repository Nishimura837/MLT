# 多項式回帰分析、k分割交差検証、optunaを使用
import pandas as pd
import optuna
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
# 評価指標のインポート
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

#test.csv,train.csvを取得
X_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_train.csv"
y_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_train.csv"
X_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/X_test.csv"
y_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/y_test.csv"
df_train_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_train.csv"
df_test_path = "/home/gakubu/デスクトップ/ML_git/MLT/ML_9/df_test.csv"
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)
df_train = pd.read_csv(df_train_path)
df_test = pd.read_csv(df_test_path)

# optunaを使ってk回交差検証を行う-------------------------
def objective(trial, x, y, cv):     #trial(探索範囲の指定用のクラス),x(説明変数),y(目的関数),k(k分割のk)
        #ハイパーパラメータごとに探索範囲を指定
        degree = trial.suggest_int('degree', 1, 10)

        #学習に使用するアルゴリズムを指定
        polynomial_features= PolynomialFeatures(degree=degree)
        
        #学習の実行、検証結果の表示
        print('Current_params : ', trial.params)
        accuracy = cross_val_score(polynomial_features, x, y, cv=cv).mean()
        return accuracy

# studyオブジェクトの作成(最大化)
study = optuna.create_study(direction='maximize')
# k分割交差検証のk
cv = 10
#目的関数の最適化
study.optimize(lambda trial: objective(trial, X_train, y_train, cv))
print('best trial')
print(study.best_trial)
print('best params')
print(study.best_params)
#---------------------------------------------------------

# 最適なハイパーパラメータを設定したモデルの定義
best_model = PolynomialFeatures(**study.best_params)
x_train_poly = best_model.fit_transform(X_train)
x_test_poly = best_model.transform(X_test)

#モデルの適合
model = LinearRegression()
model.fit(x_train_poly, y_train)
y_train_pred = model.predict(x_train_poly)
y_test_pred = model.predict(x_test_poly)


#各種評価指標をcsvファイルとして出力する
df_ee = pd.DataFrame({'R^2(決定係数)': [r2_score(y_test, y_test_pred)],
                        'RMSE(二乗平均平方根誤差)': [np.sqrt(mean_squared_error(y_test, y_test_pred))],
                        'MSE(平均二乗誤差)': [mean_squared_error(y_test, y_test_pred)],
                        'MAE(平均絶対誤差)': [mean_absolute_error(y_test, y_test_pred)]})
df_ee.to_csv("/home/gakubu/デスクトップ/ML_git/MLT/ML_9/ML_9_2/Error Evaluation 9_2.csv",encoding='utf_8_sig', index=False)
