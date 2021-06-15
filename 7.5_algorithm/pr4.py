from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

# データの読み込み
data = datasets.load_digits()
x = data.images.reshape(len(data.images), -1)
y = data.target
image = data.images

# データの分割
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# モデルの生成
model = model = MLPClassifier(hidden_layer_sizes=(500, 500,))

# 学習
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# 精度の検証
print("accuracy_score : ", accuracy_score(y_pred, y_test), "\n")
print("reliability : ")

# 信頼度の計算
#dict = {}
smvalue = np.array(model.predict_proba(X_test))
for i in range(len(y_test)):
    if y_test[i] != y_pred[i]:
        print(" num : ", i)
        print(" y_test : ", y_test[i])
        print(" y_pred : ", y_pred[i])
        print(" Softmax : ", max(smvalue[i]), "\n")
        #dict[i] = max(smvalue[i])

#num = max(dict, key=dict.get)
num = int(input("select_num : "))
print("highest_error_data : ")
print(" num : ", num)
print(" y_test : ", y_test[num])
print(" y_pred : ", y_pred[num])
print(" Softmax : ", max(smvalue[num]), "\n")

plt.imshow(X_test[num].reshape(8, 8))
plt.title(str(y_test[num]) + "to" + str(y_pred[num]) +
          "  Softmax : " + str(smvalue[num])[:5])
plt.show()
