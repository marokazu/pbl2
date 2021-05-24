# ライブラリのインポート
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
# データの読み込み
df = pd.read_csv("csv\iris.csv")
x_table = df.drop(columns="species")
y = df["species"]
x = x_table.values
# データの前処理
x = preprocessing.minmax_scale(x)
# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# モデルの定義
model = MLPClassifier(hidden_layer_sizes=(500, 500, 100,))
# 学習
model.fit(x_train, y_train)
# 予測
y_pred = model.predict(x_test)
# 精度の検証
print("ACCURACY SCORE")
print(accuracy_score(y_test, y_pred))
