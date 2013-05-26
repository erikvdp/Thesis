'''
Created on 27 apr. 2013

@author: Erik Vandeputte

Voorspellen van de muziekfactoren op de testset op basis van de MFCC-vectoren van de training set
op basis van een lineaire methode
'''
import os
import cPickle as pickle
import numpy as np

from sklearn import linear_model

MFCCDIR = '../clusterdata/featuresnpy/soft/5/'
FACTOR_FILE = './pklfiles/factormatrices.pkl'
SONG_7DIGITALID_FILE = "./pklfiles/songs_7digitalids.pkl"

TEST_FACTOR_FILE = './pklfiles/V_test_audio.pkl'


NUM_CLUSTERS = 700

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]
    

def load_data():
    print "loading data"
    global V_train,songid_ind,ind_songid,songid_digitalid
    with open(FACTOR_FILE,'r') as f:
        data = pickle.load(f)
        V_train = data['V']
        songid_ind = data['songs_map'] #(songid, songnr)
        ind_songid = data['songs_map_inv']#(songnr, songid)
    with open(SONG_7DIGITALID_FILE,'r') as f:
        songid_digitalid = pickle.load(f)

def build_x_y():
    print "building feature sets"
    global X_train,X_test
    training_size = 8000
    test_size = 1943
    
    X_train = np.zeros((training_size,NUM_CLUSTERS))
    for i in range(training_size):
        songid = ind_songid[i]
        digitalid = songid_digitalid[songid]
        f = digitalid + '_features.npy'
        data = np.load(os.path.join(MFCCDIR,f))
        features = data.sum(axis = 0)
        X_train[i] = features
    
    X_test = np.zeros((test_size,NUM_CLUSTERS))
    for i in range(test_size):
        songid = ind_songid[i+training_size]
        digitalid = songid_digitalid[songid]
        f = digitalid + '_features.npy'
        data = np.load(os.path.join(MFCCDIR,f))
        features = data.sum(axis = 0)
        X_test[i] = features


def regression():
    print "performing regression"
    global V_test
    clf = linear_model.RidgeCV(alphas=np.array([ 0.001,0.01,0.1,1,100,1000,10000,100000,1000000,10000000]))
    clf.fit(X_train,V_train)
    # The mean square error on the training set
    mse = np.mean((clf.predict(X_train) - V_train) ** 2)
    print ("training MSE: %.4f" %mse)
    
    #do prediction for the test set
    V_test = clf.predict(X_test)
    
def save_data():
    print "saving data"
    with open(TEST_FACTOR_FILE, 'w') as f:
        pickle.dump(V_test, f, pickle.HIGHEST_PROTOCOL)


load_data()
build_x_y()
regression()
save_data()    