# ライブラリのインポート
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
# データの読み込み
df = pd.read_csv("iris.csv")
y_table = df["???"]  # どれを説明変数とするか
x_table = df.drop(columns="???")  # どれを目的変数とするか
y = y_table.values
x = x_table.values
# データの前処理
x = preprocessing.minmax_scale(x)
# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# モデルの定義
model = LinearSVC()
# 学習
model.fit(???, ???)
# 予測
y_pred = model.predict(???)
# 精度の検証
print("ACCURACY SCORE")
print(accuracy_score(y_???, y_???))
