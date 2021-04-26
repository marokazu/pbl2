#3層ニューラルネット

import numpy as np

#シグモイド関数
def sigmoid(x):
    return 1/(1-np.exp(-x))

#活性化関数
def identity_function(x):
    return x

#ニューラルネット各値の定義
def init_network():
    network={}
    #ウェイト
    network['W1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['W2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['W3']=np.array([[0.1,0.3],[0.2,0.4]])
    #バイアス
    network['b1']=np.array([0.1,0.2,0.3])
    network['b2']=np.array([0.1,0.2])
    network['b3']=np.array([0.1,0.2])

    return network

#ニューラルネットによる計算
def forward(netork,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    #print("a1="+str(a1))
    z1=sigmoid(a1)
    #print("z1="+str(z1))

    a2=np.dot(z1,W2)+b2
    #print("a2="+str(a2))
    z2=sigmoid(a2)
    #print("z2="+str(z2))
    
    a3=np.dot(z2,W3)+b3
    #print("a3="+str(a3))
    y=identity_function(a3)

    return y

network=init_network()
x=np.array([1.0,5.0])
y=forward(network,x)
print(y)
