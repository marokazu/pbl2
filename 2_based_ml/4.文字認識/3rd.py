from PIL import Image
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt
import pickle
import sys
import os
os.chdir('./deep-learning-fram-scratch-master/ch03')
# os.chdir('deep-learning-fram-scratch-master\ch03')
sys.path.append(os.pardir)

# 0or1の関数


def step_function(x):
    return np.array(x > 0, dtype=np.int)

# シグモイド関数


def sigmoid(x):
    return 1/(1+np.exp(-x))

# Relu関数


def relu(x):
    return np.maximum(0, x)


def identify_function(x):
    return x

# 3層ニューラルネットワーク


def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3)+b3
    y = identify_function(a3)
    # y=softmax(a3)
    return y


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, w1)+b1
    z1 = sigmoid(a1)  # Relu
    a2 = np.dot(z1, w2)+b2
    z2 = sigmoid(a2)  # Relu
    a3 = np.dot(z2, w3)+b3
    # y=identify_function(a3)
    y = softmax(a3)
    return y

# ソフトマックス関数


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    # y=forward(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))

print("\nX="+str(len(x)))
print("X="+str(x))
print("W1="+str(len(network['W1'])))
print("W1="+str(network['W1']))
print("W2="+str(len(network['W2'])))
print("W2="+str(network['W2']))
print("W3="+str(len(network['W3'])))
print("W3="+str(network['W3']))
print("b1="+str(len(network['b1'])))
print("b1="+str(network['b1']))
print("b2="+str(len(network['b2'])))
print("b2="+str(network['b2']))
print("b3="+str(len(network['b3'])))
print("b3="+str(network['b3']))
