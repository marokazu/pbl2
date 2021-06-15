# ライブラリのインポート
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd

# データの読み込み
df = pd.read_csv("wine.csv")
y_table = df["Wine"]
x_table = df.drop(columns="Wine")
y = y_table.values
x = x_table.values
name = x_table.columns

# データの前処理
x = preprocessing.minmax_scale(x)

# モデルの定義
model = RandomForestClassifier(n_estimators=10)

# 学習
model.fit(x, y)

# 因子重要度
imp = model.feature_importances_

# 出力
out = []
out.append(name)
out.append(imp)
out = np.array(out).T
dfo = pd.DataFrame(out)
dfo.columns = ["項目", "重要度"]
print(dfo)

# 分布の可視化
dfl = df["Nonflavanoid.phenols"].values
dfh = df["Proline"].values
N1 = []
N2 = []
N3 = []
P1 = []
P2 = []
P3 = []
for i in range(len(y)):
    if y[i] == 1:
        N1.append(dfl[i])
        P1.append(dfh[i])
    elif y[i] == 2:
        N2.append(dfl[i])
        P2.append(dfh[i])
    else:
        N3.append(dfl[i])
        P3.append(dfh[i])
plt.boxplot([N1, N2, N3], labels=["1", "2", "3"])
plt.title("Nonflavanoid.phenols[importance=LOW]")
plt.show()
plt.boxplot([P1, P2, P3], labels=["1", "2", "3"])
plt.title("Proline[importance=HIGH]")
plt.show()
