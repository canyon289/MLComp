"""
Takes pickled model and shows feature breakdown
"""

# Load in pickled model
model = pickle.load(open("xgb_ntrees_1_depth_1_eta_0.3.p", 'rb'))

