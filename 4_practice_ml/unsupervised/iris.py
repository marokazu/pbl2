# ライブラリのインポート
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import matplotlib.pyplot as plt

# データの読み込み
df = pd.read_csv("iris.csv")
y = df["species"].values
df = df.drop(columns=["species"])
df = df.values

# デンドログラム作成
z = linkage(df, method="median", metric="euclidean")
dendrogram(z, labels=y)
plt.show()

# 次元圧縮
model = PCA(n_components=2)
model.fit(df)
x = model.transform(df)

# k-means クラスタリング
model2 = KMeans(n_clusters=3)
model2.fit(x)
ypred = model2.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=ypred, cmap="brg")
plt.show()
