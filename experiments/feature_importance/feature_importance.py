"""
Build tree and see what i can understand from feature importance
"""

import pickle
import xgboost as xgb
from matplotlib import pyplot

model = pickle.load(open("clf.p", 'rb'))

clf = model['clf']
v = model['vectorizer']
clf.get_booster().feature_names = v.get_feature_names()
xgb.to_graphviz(clf, num_trees=1).render("MLCompSingleLaptop.gv", view=True)
xgb.plot_importance(clf, max_num_features = 10, )
pyplot.show()
