# ライブラリのインポート
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# データの読み込み
df = pd.read_csv("rent.csv", encoding="shift-jis")
y_table = df["???"]
x_table = df.drop(columns="???")
y = y_table.values
x = x_table.values
# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# モデルの定義
reg = linear_model.LinearRegression()
# 学習
???.fit(???, ???)
# 予測
y_pred = ???.predict(???)
# 精度の検証
print("決定係数")
print(reg.score(x, y))
print()
print("ROOT MEAN SQUARED ERROR")
print(np.sqrt(mean_squared_error(y_???, y_???)))
