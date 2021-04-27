# In[1]
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# In[2]
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
plt.show()

# In[3]
model = GaussianNB()
model.fit(X, y)

# In[4]
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

# In[5]
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
plt.show()

# In[6]
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)
print(yprob[-8:].round(2))

# In[7]
sns.set()
data = fetch_20newsgroups()
data.target_names
print(data.target_names)

# In[8]
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# In[9]
print(train.data[5])

# In[10]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# In[11]
model.fit(train.data, train.target)
labels = model.predict(test.data)

# In[12]
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# In[13]


def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]


# In[14]
print(predict_category('sending a payload to the ISS'))

# In[15]
print(predict_category('discussing islam vs atheism'))

# In[16]
print(predict_category('determining the screen resolution'))
