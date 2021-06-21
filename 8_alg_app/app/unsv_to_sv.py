import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

#データの読み込み
Xdata, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)

#データの前処理
Xdata = preprocessing.minmax_scale(Xdata)

#クラスタリング前のデータを可視化
plt.scatter(Xdata[:, 0], Xdata[:, 1], s=50)
plt.show()

#クラスタリングを定義
kmeans = KMeans(n_clusters=4)

#学習
kmeans.fit(Xdata)

#ラベリング
y_kmeans = kmeans.predict(Xdata)

#ラベリングの確認の可視化
plt.scatter(Xdata[:, 0], Xdata[:, 1], c=y_kmeans, s=50, cmap='brg')
plt.show()

#SVMを定義
model=LinearSVC()

#学習
model.fit(Xdata,y_kmeans)

#決定境界の可視化
