'''
Created on 2 nov. 2012

@author: Erik Vandeputte
'''
from sklearn import svm
from sklearn import cross_validation
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_score
import pylab as pl
import numpy as np
import cPickle as pickle
import time

NUM_ITERATIONS = 10

ECHONEST_FILE = "../../../msd_dense_subset/mood_echonest_features.pkl" #collection of features for the small dataset
ECHONEST_FILE_2 = "../../../msd_dense_subset/mood_echonest_features_2.pkl" #collection of features for the big dataset

MFCC_FILE = "../../../msd_dense_subset/mood_mfcc_features.pkl" #collection of features for the small dataset
MFCC_FILE_2 = "../../../msd_dense_subset/mood_mfcc_features_2.pkl" #collection of features for the big dataset


CLASSIFIER_FILE = "../../../msd_dense_subset/classifier" #dump of the classifier that can be used in mood_detection_application

def train_classifier():
#this method is used to train the classifier on dataset 2
    print 'training data'
    svc = RandomForestClassifier(n_estimators=500) 
    with open(ECHONEST_FILE_2, 'r') as f:
            data = pickle.load(f)
            labels = data['labels']
            features = data['features']
    svc.fit(features,labels)
    data = {
            'classifier': svc
            }
    with open(CLASSIFIER_FILE, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    print "data saved to %s" % CLASSIFIER_FILE
    
#2testing the features using cross validation
def cross_val(data_file , svc = None):
    if svc == None:
        svc = svm.SVC(kernel='rbf', C=1)
    print "loading data"
    with open(data_file, 'r') as f:
            data = pickle.load(f)
            labels = data['labels']
            features = data['features']
    if (type(svc) == svm.SVC):
        #standardization important for SVM's!
        features = preprocessing.scale(features)
    start_time = time.time()
    print "starting cross validation"
    scores =  cross_validation.cross_val_score(svc, features, labels, cv=10, n_jobs=-1)
    print (scores.mean(), scores.std() / 2)
    print "Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    print "Runtime : %.2f seconds" % (time.time() - start_time)
    

def optimize_par(data_file):
    #this function optimizes hyperparameters for different classifiers
    with open(data_file, 'r') as f:
            data = pickle.load(f)
            labels = data['labels']
            features = data['features']
    #par ={'C': [0.1, 1, 10, 100, 1000], 'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    #par ={'C': [0.1, 1, 10, 100, 1000], 'degree': range(5), 'kernel': ['poly']}
    #par ={'n_neighbors' : [1,2,3,4,5,6,7,8,9,10]}
    par ={'n_estimators' : [50,75,100,150,200,300,400,500,750]}
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=0)
    scores = [('error_rate', zero_one_score),]
    
    for score_name, score_func in scores:
        print "# Tuning hyper-parameters for %s" % score_name
        print
        #classifier = svm.SVC()
        #classifier = neighbors.KNeighborsClassifier(weights='distance')
        classifier = RandomForestClassifier()
        clf = GridSearchCV(classifier, par, score_func=score_func)
        clf.fit(X_train, y_train, cv=5)
        print "Best parameters set found on development set:"
        best_parameters, score,_ = max(clf.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(par.keys()):
            print "%s: %r %r" % (param_name, best_parameters[param_name],score)
            

def visualize(data_file):
#try to viszualize using PCA to reduce the dimension of the feature space to 2
    np.set_printoptions(threshold='nan')
    with open(data_file, 'r') as f:
        data = pickle.load(f)
        labels = data['labels']
        features = data['features']
    features = preprocessing.scale(features)
    pca = decomposition.PCA(n_components=2)
    X = pca.fit_transform(features) #Fit the model with X and apply the dimensionality reduction on X.
    positivemean = np.mean(X[labels == 0], axis = 0)
    negativemean = np.mean(X[labels == 1], axis = 0)
    pl.show()
    pl.figure()
    #pl.plot(positivemean[0] ,positivemean[1],"o" ,color = 'green' , ms=12)
    #pl.plot(negativemean[0] ,negativemean[1],"o" ,color = 'red' , ms=12)
    pl.scatter(X[labels == 0, 0], X[labels == 0, 1],marker = '+',color ='green', label="positieve emotie")
    pl.scatter(X[labels == 1, 0], X[labels == 1, 1],marker = '+',color='red', label="negatieve emotie")
    
    pl.title('PCA op de akoestische features')
    pl.xlabel('1e principale component')
    pl.ylabel('2e principale component')
    pl.legend()
    pl.show()

def visualize_2(data_file):
    np.set_printoptions(threshold='nan')
    with open(data_file, 'r') as f:
        data = pickle.load(f)
        labels = data['labels']
        features = data['features']
        
    X = np.column_stack((features[:,0],features[:,3]))
    positivemean = np.mean(X[labels == 0], axis = 0)
    negativemean = np.mean(X[labels == 1], axis = 0)
    pl.show()
    pl.figure()
    #pl.plot(positivemean[0] ,positivemean[1],"o" ,color = 'green' , ms=12)
    #pl.plot(negativemean[0] ,negativemean[1],"o" ,color = 'red' , ms=12)
    pl.scatter(X[labels == 0, 0], X[labels == 0, 1],marker = '+',color ='green', label="positieve emotie")
    pl.scatter(X[labels == 1, 0], X[labels == 1, 1],marker = '+',color='red', label="negatieve emotie")
    
    pl.xlabel('snelheid van de aanzet')
    pl.ylabel('gemiddeld volume')
    pl.legend()
    pl.show()


def run_analyzer(data_file,classifier = None):

    if classifier == None:
        classifier = RandomForestClassifier(n_estimators=75)
    start_time = time.time()
    with open(data_file, 'r') as f:
        data = pickle.load(f)
        labels = data['labels']
        features = data['features']
    num_labels = len(np.unique(labels))
    
    avg_classifications = np.zeros((num_labels,num_labels))
    for j in range(NUM_ITERATIONS):
        #split into training and test data
        training_features, test_features, training_labels, test_labels = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=0)
        
        classifier.fit(training_features, training_labels)
        
        predictions = classifier.predict(test_features)
        predictions = map(float, predictions)
        test_labels = map(float, test_labels)
        succes_rate = float(np.count_nonzero(np.equal(predictions, test_labels)))/len(test_labels)
        
        classifications = np.zeros((num_labels,num_labels))
        for i in range(len(test_labels)):
            classifications[test_labels[i],predictions[i]] = classifications[test_labels[i],predictions[i]] + 1
        #express classifications in relative terms
        for i in range(num_labels):
            classifications[i,:] = classifications[i,:] / float(np.count_nonzero(np.equal(test_labels, i)))
        avg_classifications = avg_classifications+ classifications
        print "results single run:"
        print "succes rate: %s" %succes_rate
        print "Runtime : %.2f seconds" % (time.time() - start_time)
        #print "number of test samples: %d" %len(test_labels)
        #print "number of happy songs: %d" %np.count_nonzero(np.equal(test_labels, 0))
        #print "number of sad songs: %d" %np.count_nonzero(np.equal(test_labels,1))
        #print "number of fear songs: %d" %np.count_nonzero(np.equal(test_labels,2))
        #print "classifications"
    
    print avg_classifications / NUM_ITERATIONS
        
##SCRIPT
#train_classifier()
#run_analyzer(ECHONEST_FILE_2)
#cross_val(MFCC_FILE)
#cross_val(ECHONEST_FILE)
#optimize_par(ECHONEST_FILE)
visualize_2(ECHONEST_FILE)