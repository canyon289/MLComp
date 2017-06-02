"""
Build tree and see what i can understand from feature importance
"""

import pickle
import xgboost as xgb
from matplotlib import pyplot
from textclassifier import MLComp


m = MLComp()

params = {'max_depth':100,
        'learning_rate':0.:,
            'n_estimators':1,
            'silent':False,
            'objective':"multi:softprob",
            #'num_class':20,
            #'booster':'gbtree',
            'n_jobs':8,
            #'nthread':4,
            'gamma':0, 
            'min_child_weight':1,
            'max_delta_step':0,
            'subsample':1,
            'colsample_bytree':1,
            'colsample_bylevel':1,
            'reg_alpha':0,
            'reg_lambda':1,
            'scale_pos_weight':1,
            'base_score':0.5,
            #'random_state':0,
            'seed':0,
            'missing':None
            }

m.run_xgb(params)
m.pickle_xgb("classifers/", "trees/")

