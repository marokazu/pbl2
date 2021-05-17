# ライブラリのインポート
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# データの読み込み
df = pd.read_csv("rent.csv", encoding="shift-jis")
y_table = df["家賃"]
x_table = df.drop(columns="家賃")
y = y_table.values
x = x_table.values
# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# モデルの定義
model = linear_model.LinearRegression()
# 学習
model.fit(x_train, y_train)
# 予測
y_pred = model.predict(x_test)
# 精度の検証
print("ROOT MEAN SQUARED ERROR")
print(mean_squared_error(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
