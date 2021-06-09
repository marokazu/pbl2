# ライブラリのインポート
import pandas
import numpy
from sklearn import linear_model

# データの読み込み
"df"
"y_table"
"x_table"
"y"
"x"
"name"

# モデルの定義
model = linear_model.LinearRegression()

# 学習


# モデル
a = model.coef_  # 重み
b = model.intercept_  # 切片

# 予測値の生成
"y_pred"

# 標準誤差を算出
se = np.sum((y-y_pred)**2, axis=0)
se = se/(x.shape[0]-x.shape[1]-1)
s = np.linalg.inv(np.dot(x.T, x))
stder = np.sqrt(np.diagonal(se*s))

# t 値を算出
t = "回帰係数"/stder

# 出力
out = {}
out.append(name)
out.append(a)
out.append(t)
out = np.array(out).T
dfo = pd.DataFrame(out)
dfo.columns = ["因子名", "重み", "t 値"]
print(dfo)
print("決定係数\t"+model.score(x, y)))
print("切片\t"+b)
