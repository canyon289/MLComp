"""
========================================================
Classification of text documents: using a MLComp dataset
========================================================

This is an example showing how the scikit-learn can be used to classify
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

The dataset used in this example is the 20 newsgroups dataset and should be
downloaded from the http://mlcomp.org (free registration required):

  http://mlcomp.org/datasets/379

Once downloaded unzip the archive somewhere on your filesystem.
For instance in::

  % mkdir -p ~/data/mlcomp
  % cd  ~/data/mlcomp
  % unzip /path/to/dataset-379-20news-18828_XXXXX.zip

You should get a folder ``~/data/mlcomp/379`` with a file named ``metadata``
and subfolders ``raw``, ``train`` and ``test`` holding the text documents
organized by newsgroups.

Then set the ``MLCOMP_DATASETS_HOME`` environment variable pointing to
the root folder holding the uncompressed archive::

  % export MLCOMP_DATASETS_HOME="~/data/mlcomp"

Then you are ready to run this example using your favorite python shell::

  % ipython examples/mlcomp_sparse_document_classification.py

"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD 3 clause


from time import time
import sys
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.datasets import load_mlcomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import pickle

import ipdb
from pprint import pprint

MLCOMP = os.path.abspath(os.path.join(__file__, '..', 'data'))

# print(__doc__)

class MLComp:

    def __init__(self):
        """ML Comp training class.
        Decided on class so I can introspect each variable and see what
        it's doing
        """
        self.load_data()
        self.extract_train_features()
        self.extract_test_features()
        self.run_xgb()
        return

    def run_xgb(self):
        from xgboost.sklearn import XGBClassifier
        parameters = {'max_depth':100,
                    'learning_rate':0.3,
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
        self.benchmark(XGBClassifier, parameters, 'XGBoost Classifier')
        return 

    def load_data(self):
        """Load the news_train file to see how it's structured"""
        # Load the training set
        print("Loading 20 newsgroups training set... ")
        self.news_train = load_mlcomp('20news-18828', 'train', mlcomp_root=MLCOMP)
        print(self.news_train.DESCR)
        print("%d documents" % len(self.news_train.filenames))
        print("%d categories" % len(self.news_train.target_names))
        
        return self.news_train
    
    def extract_train_features(self): 
        print("Extracting train features from the dataset using a sparse vectorizer")
        t0 = time()
        # I can instantiate the vectorize object without needing to pass in anything
        self.vectorizer = TfidfVectorizer(encoding='latin1')
        
        # See what the input files look like
        # They're just strings. The vectorizer automatically splits them apart and does some processing
        train_input_strings = list(open(f, encoding='latin-1').read() for f in self.news_train.filenames)

        self.X_train = self.vectorizer.fit_transform(train_input_strings)
        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % self.X_train.shape)

        assert sp.issparse(self.X_train)
        self.y_train = self.news_train.target
        return 

    def extract_test_features(self):

        print("Loading 20 newsgroups test set... ")
        self.news_test = load_mlcomp('20news-18828', 'test', mlcomp_root=MLCOMP)
        t0 = time()
        print("done in %fs" % (time() - t0))

        print("Predicting the labels of the test set...")
        print("%d documents" % len(self.news_test.filenames))
        print("%d categories" % len(self.news_test.target_names))

        print("Extracting features from the dataset using the same vectorizer")
        t0 = time()
        
        # Read in test data
        test_input_strings = (open(f, encoding='latin-1').read() for f in self.news_test.filenames)
        self.X_test = self.vectorizer.transform(test_input_strings)
        self.y_test = self.news_test.target
        print("done in %fs" % (time() - t0))
        print("n_samples: %d, n_features: %d" % self.X_test.shape)
        return

    def run_sgd(self):
        print("Testbenching a linear classifier...")
        parameters = {
            'loss': 'hinge',
            'penalty': 'l2',
            'n_iter': 50,
            'alpha': 0.00001,
            'fit_intercept': True,
        }

        self.benchmark(SGDClassifier, parameters, 'SGD')

    def run_multinominal(self):
        print("Testbenching a MultinomialNB classifier...")
        parameters = {'alpha': 0.01}

        self.benchmark(MultinomialNB, parameters, 'MultinomialNB')

        return

    def benchmark(self, clf_class, params, name):
        """Benchmark classifiers"""
        
        print("parameters:", params)
        t0 = time()
        clf = clf_class(**params).fit(self.X_train, self.y_train)
        pickle.dump(clf, open("../clf.p", "wb"))
        print("done in %fs" % (time() - t0))

        if hasattr(clf, 'coef_'):
            print("Percentage of non zeros coef: %f"
                  % (np.mean(clf.coef_ != 0) * 100))

        print("Predicting the outcomes of the testing set")
        t0 = time()
        pred = clf.predict(self.X_test)
        print("done in %fs" % (time() - t0))

        print("Classification report on test set for classifier:")
        print(clf)
        print()
        print(classification_report(self.y_test, pred,
                                    target_names=self.news_test.target_names))

        cm = confusion_matrix(self.y_test, pred)
        print("Confusion matrix:")
        print(cm)

        # Show confusion matrix
        plt.matshow(cm)
        plt.title('Confusion matrix of the %s classifier' % name)
        plt.colorbar()

        plt.show()

