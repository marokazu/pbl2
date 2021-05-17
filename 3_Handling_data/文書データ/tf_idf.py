from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

np.set_printoptions(threshold=3)

# データのロード
category = ["talk.religion.misc",
            "soc.religion.christian", "sci.space", "comp.graphics"]
text = fetch_20newsgroups(
    remove=("headers", "footers", "quotes"), categories=category)
# TF-IDF 値作成
tivec = TfidfVectorizer(max_df=0.9)
x = tivec.fit_transform(text.data)
x = x.toarray()

print(x)
