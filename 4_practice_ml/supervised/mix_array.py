# ライブラリのインポート
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.metrics import precision_score, f1_score, classification_report

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

# 混合行列
print(confusion_matrix(y_test, y_pred))
print()

# 精度の検証
print("ACCURACY SCORE")
print(accuracy_score(y_test, y_pred))
print()

# 再現率
print("Recall")
print(recall_score(y_test, y_pred, average="macro"))
print()

# 適合率
print("Precision")
print(precision_score(y_test, y_pred, average="macro"))
print()

# F値
print("F1-score")
print(f1_score(y_test, y_pred, average="macro"))
print()

# 分類結果
print(classification_report(y_test, y_pred))
