"""
Test to see if I can combine sparse matrices with additional feature engineering
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np
from scipy.sparse import hstack

train_set = ["This is a positive sentence 7",
            "This is a negative sentence 8"]

transformer = TfidfVectorizer()    
sp = transformer.fit_transform(train_set)


df = pd.DataFrame(train_set, columns=['text'])
df["contains7"] = df["text"].str.contains("7")
df["contains8"] = df["text"].str.contains("7")
df2 = pd.DataFrame([[10,12],[11,13]], columns = ('a','b'))

c = np.matrix(df["contains7"]).T
n = np.array([[1,2,3],[4,5,6]])


# Test Sparse Stack
# Scipy HStack works but npstack doesnt
stack = hstack((sp,n))
