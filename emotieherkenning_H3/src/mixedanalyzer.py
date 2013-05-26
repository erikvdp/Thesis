'''
Created on 20 nov. 2012

@author: Erik Vandeputte
'''
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import decomposition
import pylab as pl
import numpy as np
import cPickle as pickle
import time

DATA_FILE = "../../msd_dense_subset/mood.pkl"

#2testing the features using cross validation
def cross_val():
    print "loading data"
    with open(DATA_FILE, 'r') as f:
            data = pickle.load(f)
            labels = data['labels']
            features = data['features']
    start_time = time.time()
    svc = svm.SVC(kernel='linear')
    svc = RandomForestClassifier(n_estimators=400)
    print "starting cross validation"
    scores =  cross_validation.cross_val_score(svc, features, labels, cv=5, n_jobs=-1)
    print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    print "Runtime : %.2f seconds" % (time.time() - start_time)
#try to viszualize using PCA to reduce the dimension of the feature space to 2
def visualize(training_features,labels):
    pca = decomposition.PCA(n_components=2)
    pca.fit(training_features)
    X = pca.transform(training_features)
    X = training_features
    pl.show()
    pl.figure()
    pl.scatter(X[labels == 1, 0], X[labels == 1, 1],c='r', label="happy")
    pl.scatter(X[labels == -1, 0], X[labels == -1, 1],c='b', label="sad")
    pl.legend()
    pl.show()

def run_analyzer():

    start_time = time.time()
    with open(DATA_FILE, 'r') as f:
        data = pickle.load(f)
        labels = data['labels']
        features = data['features']

    #split into training and test data
    training_features, test_features, training_labels, test_labels = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=0)
    
    svc = svm.SVC(kernel='linear')
    svc.fit(training_features, training_labels)
    
    predictions = svc.predict(test_features)
    predictions = map(float, predictions)
    test_labels = map(float, test_labels)
    test_labels = np.array(test_labels)
    results = predictions * test_labels #if results[i] == -1 => false prediction
    succes_rate = float(len(np.where(results == 1)[0]))/len(test_labels)
    
    print "results single run:"
    print "succes rate: %s" %succes_rate
    print "Runtime : %.2f seconds" % (time.time() - start_time)

##SCRIPT
#run_analyzer()
cross_val()
#visualize(training_features,training_labels)