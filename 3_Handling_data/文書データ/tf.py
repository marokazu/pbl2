from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
np.set_printoptions(threshold=np.inf)

#データのロード
#category = ["talk.religion.misc", "soc.religion.christian", "sci.space", "comp.graphics"]
category = ["talk.religion.misc"]
text = fetch_20newsgroups(remove = ("headers", "footers", "quotes"), categories = category)

#TF 値作成
tfvec = CountVectorizer(max_df = 0.9)
tfvec.fit(text.data)

#print(tfvec.get_feature_names())
#print(len(tfvec.get_feature_names()))
x = tfvec.transform(text.data)

#特徴量ベクトルに変換（出現頻度）
vector = x.toarray()
"""
for i in range(len(vector)):
    print(text.data[i])
    print(vector[i])
"""
print(len(vector[0]))
#print(vector)
