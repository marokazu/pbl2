from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#データの読み込み
df=pd.read_csv("wine.csv")
x_table=df.drop(columns="Wine")
x=x_table.values
name=x_table.columns

#データの前処理
x = preprocessing.minmax_scale(x)
#モデルの定義
pca = PCA(n_components=len(x[0]))
#学習
pca.fit(x)
#寄与率
con=pca.explained_variance_ratio_
index=[]
for i in range(len(con)):
    index.append("第"+str(i+1)+"主成分")
dfc=pd.DataFrame(con)
dfc.columns=["寄与率"]
dfc.index=index
print(dfc)

#固有ベクトル
fac=pca.components_

#因子負荷量
factor=[]
for i in range(len(con)):
    factor.append(np.sqrt(con[i])*fac[i])
dfo=pd.DataFrame(factor)
dfo.columns=name
dfo.index=index
print(dfo)

#matplotlib
#第1,2,3主成分軸
tx=pca.transform(x)
fig = plt.figure()

#3D
ax1 = fig.add_subplot(2, 2, 2, projection='3d', adjustable='box')
ax1.set_xlabel('PCA1')
ax1.set_ylabel('PCA2')
ax1.set_zlabel('PCA3')
ax1.scatter(tx[:,0],tx[:,1],tx[:,2])

#2D 1-2
ax2 = fig.add_subplot(2, 2, 1, adjustable='box')
ax2.set_xlabel('PCA1', labelpad=100)
ax2.set_ylabel('PCA2', labelpad=160)
ax2.scatter(tx[:,0],tx[:,1])
ax2.spines['left'].set(position='zero')
ax2.spines['bottom'].set(position='center')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

#2D 1-3
ax3 = fig.add_subplot(2, 2, 3, adjustable='box')
ax3.set_xlabel('PCA1', labelpad=100)
ax3.set_ylabel('PCA3', labelpad=160)
ax3.scatter(tx[:,0],tx[:,2])
ax3.spines['left'].set(position='zero')
ax3.spines['bottom'].set(position='center')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

#2D 2-3
ax4 = fig.add_subplot(2, 2, 4, adjustable='box')
ax4.set_xlabel('PCA2', labelpad=100)
ax4.set_ylabel('PCA3', labelpad=160)
ax4.scatter(tx[:,1],tx[:,2])
ax4.spines['left'].set(position='zero')
ax4.spines['bottom'].set(position='center')
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

plt.show()

#第11,12,13主成分軸