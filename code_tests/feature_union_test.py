"""
Test to see if I can combine sparse matrices with additional feature engineering
"""

import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin

train_set = ["This is a positive sentence 7",
            "This is a negative sentence 8"]

df = pd.DataFrame(train_set, columns=['text'])

class HandBuiltFeatures():
    def __init__(self):
        self.feature_names = None
        return
    
    def fit(self, x, y=None):
        """This transformer is stateless
        Fit returns itself after its been modified by the data but I don't
        have that in this case so nothing needed
        """ 
        return self
    
    def transform(self, raw_df):
        """This is where the magic happens"""
        transformed_df = pd.DataFrame()
        columns = ["contains7", "contain8"]
        transformed_df["contains7"] = raw_df["text"].str.contains("7")
        transformed_df["contains8"] = raw_df["text"].str.contains("8")
        
        array = transformed_df.values()
        self.features = transformed_df.colummns

    def get_feature_names()
        if self.features is None:
            return ValueError("Features not fitted yet")
        return self.features

"""
class dftfidVectorizer(TfidfVectorizer)
    def __init__():
        super().__init__()

    def fit(self, df, y=None):
        text = df["text"]
        return super
"""

transformer = TfidfVectorizer()    
sp = transformer.fit_transform(train_set)


c = np.matrix(df["contains7"]).T
n = np.array([[1,2,3],[4,5,6]])

