# ライブラリのインポート
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report

# データの読み込み
df = pd.read_csv("iris.csv")
y_table = df["species"]
x_table = df.drop(columns="species")
y = y_table.values
x = x_table.values
# データの前処理
x = preprocessing.minmax_scale(x)
# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# モデルの定義
model = LinearSVC()
# 学習
model.fit(x_train, y_train)
# 予測
y_pred = model.predict(x_test)

# 分類結果
print(classification_report(y_test, y_pred))
