'''
Created on 15 nov. 2012
This module tries to detect the mood of a song using lyric information
@author: Erik Vandeputte
'''
import time
import cPickle as pickle
from sklearn.naive_bayes import MultinomialNB
import sklearn.svm
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import zero_one_score
import numpy as np

DATA_FILE = "../../msd_dense_subset/mood_lyrics_features.pkl"

DATA_FILE_2 = "../../msd_dense_subset/mood_lyrics_features2.pkl"


#2testing the features using cross validation
def cross_val(data_file,classifier = None):
    if classifier == None:
        classifier = svm.LinearSVC(C = 0.5)
    #print "loading data"
    with open(data_file, 'r') as f:
            data = pickle.load(f)
            labels = data['labels']
            features = data['features']
    start_time = time.time()
    print "cross validation"
    scores =  cross_validation.cross_val_score(classifier, features, labels, cv=10, n_jobs=-1)
    print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    print "Runtime : %.4f seconds" % (time.time() - start_time)

def search_parameters(data_file):
    with open(data_file, 'r') as f:
            data = pickle.load(f)
            labels = data['labels']
            features = data['features']
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    features, labels, test_size=0.5, random_state=0)
    scores = [
    ('error_rate', zero_one_score),]
    
    #classifier = svm.LinearSVC()
    classifier = MultinomialNB()
    
    tuned_parameters = {'alpha' :(0.001, 0.01,0.1,0.5,1,1.5,2,5,10) }
    #tuned_parameters = {'C' :(0.00001, 0.001, 0.01, 0.1,0.5,1,1.5,2,5,10,20,50,100,500,1000)}
    for score_name, score_func in scores:
        print "# Tuning hyper-parameters for %s" % score_name
        print
    
        clf = GridSearchCV(classifier, tuned_parameters, score_func=score_func)
        clf.fit(X_train, y_train, cv=5)
    
        print "Best parameters set found on development set:"
        best_parameters, score,_ = max(clf.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(tuned_parameters.keys()):
            print "%s: %r" % (param_name, best_parameters[param_name])
            
def run_analyzer(data_file):
    start_time = time.time()
    with open(data_file, 'r') as f:
                data = pickle.load(f)
                labels = data['labels']
                features = data['features']
    
    #split into training and test data
    training_features, test_features, training_labels, test_labels = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=0)
    
    
    clf = svm.SVC()
    clf.fit(training_features, training_labels)
    clf = MultinomialNB().fit(training_features, training_labels)
    print "number of training samples %d" %len(training_labels)
    print "number of test samples: %d" %len(test_labels)
    print "number of features: %d" %training_features.shape[1]
    print "score on the training data: %.2f: " %clf.score(training_features, training_labels)
    predictions = clf.predict(test_features)
    predictions = map(float, predictions)
    test_labels = map(float, test_labels)
    test_labels = np.array(test_labels)
    succes_rate = np.mean(predictions == test_labels)
    
    print "results fitting on test data:"
    print "succes rate: %s" %succes_rate
    print "Runtime : %.2f seconds" % (time.time() - start_time)

##SCRIPT
#run_analyzer(DATA_FILE_2)
#cross_val(DATA_FILE)
#cross_val(DATA_FILE_2)
#search_parameters(DATA_FILE_2)