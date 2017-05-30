"""
Test combining sparse features with other features and see what I can get
out of xgboost
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# Setup some fake data
# The words positive and negative should be important features
train_set = ["This is a positive sentence 7",
            "This is a negative sentence 8"]

train_targets = np.array([1,0])
test_set = [["Also a positive sentence 7"]]


# Vectorize 
transformer = TfidfVectorizer()    
sparse_featureset = transformer.fit_transform(train_set)
df_features = pd.DataFrame(sparse_featureset.todense(), columns=transformer.get_feature_names())

# Add another feature
contains_7 = pd.Series([int(("7" in s)) for s in train_set])
df_features["Contains7"] = contains_7


# SKLearn API
cls = XGBClassifier(silent=True)

cls.fit(X = df_features, y=train_targets)
print(cls.booster().get_fscore())

df_features = df_features.drop(df_features.columns[1], axis=1)
train_data = xgb.DMatrix(df_features.values, label=train_targets)

# Generic parameters
param = {'max_depth':5,
        #'objective':'reg:linear',
        'objective':'multi:softprob','num_class':2,
        'eta':.3,
        'silent':0,
        'colsample_bytree':.2,
        'nround':100}

m = xgb.train(params = param, dtrain = train_data)
print(df_features.columns)
print(m.get_fscore())
xgb.to_graphviz(m, num_trees=2)
