import numpy as np
import matplotlib.pylab as plt
# AND回路


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# OR回路


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# NAND回路


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# NOR回路


def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# XOR回路


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# ステップ関数


def step_function(x):
    y = np.array(x > 0, dtype=np.int)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
    return y

# シグモイド関数


def sigmoid(x):
    y = 1/(1+np.exp(-x))
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()
    return y

# ReLU関数


def RELU(x):
    y = np.maximum(0, x)
    plt.plot(x, y)
    plt.ylim(-0.1, 5.5)
    plt.show()
    return y


x = np.arange(-5, 5, 0.1)
sigmoid(x)

x = np.arange(-5, 5, 0.1)
RELU(x)

x = np.arange(-5, 5, 0.1)
step_function(x)
