"""
Takes pickled model and shows feature breakdown
"""
import sys
import matplotlib.pyplot as plt
import pickle
import xgboost

# Load in pickled model
file_name = sys.argv[1]
print(file_name)
d_clf= pickle.load(open(file_name, 'rb'))

plt.matshow(d_clf["confusion_matrix"])
plt.colorbar()
plt.show()

