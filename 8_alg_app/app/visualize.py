#ライブラリのインポート
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#データの読み込み
df=pd.read_csv("iris.csv")
y=df["category"].values
x=df.drop(columns="category").values

#次元圧縮
model=PCA(n_components=2)
model.fit(x)
x=model.transform(x)

#データの前処理 
x=preprocessing.minmax_scale(x)

#モデルの定義
model=RandomForestClassifier(n_estimators=10)

#学習
model.fit(x,y)

#可視化
fig, ax = plt.subplots(figsize=(8, 6))
X, Y = np.meshgrid(np.linspace(*ax.get_xlim(), 1000), np.linspace(*ax.get_ylim(), 1000))
XY = np.column_stack([X.ravel(), Y.ravel()])
Z = model.predict(XY).reshape(X.shape)
plt.contourf(X, Y, Z, alpha=0.1, cmap='brg')
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='brg')
plt.xlim(min(x[:,0]),max(x[:,0]))
plt.ylim(min(x[:,1]),max(x[:,1]))
plt.show()