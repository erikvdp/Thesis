'''
Created on 4 apr. 2013

@author: Erik Vandeputte
'''
import cPickle as pickle
import numpy as np
import os
import time

from sklearn import linear_model
from sklearn import cross_validation
from sklearn import preprocessing

INTERACTION_MATRIX_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
SONG_FACTORS_FILE = '../../msd_dense_subset/song_factors_200.pkl'
SONG_7DIGITALID_FILE = "../pklfiles/songs_7digitalids.pkl"

FACTORS_FILE_1 ="../pklfiles/new_V_original_random.pkl"
FACTORS_FILE_2 ="../pklfiles/new_V_random_random.pkl"
FACTORS_FILE_3 ="../pklfiles/new_V_prediction_random.pkl"

TEST_SONGS_FILE="../pklfiles/test_songs.pkl"

NUM_FACTORS = 200
NUM_FRAMES = 5

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def load_data():
    global V, U,songid_ind, ind_songid,song_7digitalid, digitalid_song,songid_least_to_most
    with open(SONG_FACTORS_FILE, 'r') as f:
        data = pickle.load(f)
        V = data['V']
        U = data['U']
        songid_ind = data['songs_map'] #(songid, songnr)
        ind_songid = data['songs_map_inv']#(songnr, songid)
    with open(SONG_7DIGITALID_FILE, 'r') as f:
        song_7digitalid = pickle.load(f)
        digitalid_song = {v:k for k, v in song_7digitalid.items()}
    with open(INTERACTION_MATRIX_FILE, 'r') as f:
        data = pickle.load(f)
        songid_least_to_most = data['songid_least_to_most']
    
        
def build_x_y(features_dir,hard,num_clusters):
    global X,y,V_ind,train_idx,test_idx
    train_idx = list()
    test_idx = list()
    V_ind = list()
    test_songids = set(songid_least_to_most[-2500:]) #25% most played songs in the test set
    files = mylistdir(features_dir)
    X = np.empty((len(files),num_clusters))
    y = np.empty((len(files),NUM_FACTORS))
    for index,f in enumerate(files):
        data = np.load(os.path.join(features_dir,f))
        if(hard): #data is a (NUM_SAMPLES,1) matrix with as value its closest centroid
            features =np.eye(num_clusters)[data]
            features =features.sum(axis = 0)
        else:
            features = data.sum(axis = 0)
        X[index] = features
        digitalid = f[0:f.find('_')]
        songid = digitalid_song[digitalid] #the original songid
        ind = songid_ind[songid] #ind = the index of the song in the original V matrix
        y[index] = V[ind]
        V_ind.append(ind)
        if(songid in test_songids):
            test_idx.append(index)
        else:
            train_idx.append(index)
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    V_ind = np.array(V_ind) #V_ind (entry in X, entry in V)

def make_random_test_set():
    global train_idx, test_idx,V_ind_test_idx
    rs = cross_validation.ShuffleSplit(X.shape[0], n_iter=1,test_size=.25, random_state=0)
    for train_index,test_index in rs:
        train_idx = train_index
        test_idx = test_index        
     
def regression():
    global predicted_factors,true_factors
    make_random_test_set()
    clf = linear_model.Ridge()
    clf.fit(X[train_idx],y[train_idx])
    # The mean square error
    mse = np.mean((clf.predict(X[train_idx]) - y[train_idx]) ** 2)
    print ("training MSE: %.4f" %mse)
    # The mean square error
    y_predictions = clf.predict(X[test_idx])
    mse = np.mean((y_predictions - y[test_idx]) ** 2)
    print ("test MSE: %f" %mse)
    predicted_factors = preprocessing.normalize(y_predictions, norm='l2')#normalize song factors
    true_factors = y[test_idx]
    return mse

def new_test_idx_to_songs():
    global test_songs
    test_songs = set()
    for i,idx in enumerate(V_ind_test_idx):
        test_songs.add(ind_songid[idx])

#replace the old songfactors with the new predictions
def replace():
    global V_1,V_2,V_3,V_ind_test_idx
    V_ind_test_idx = V_ind[test_idx]
    V_1 = np.copy(V)
    V_2 = np.copy(V)
    V_3 = np.copy(V)
    V_3[V_ind_test_idx] = predicted_factors
    np.random.shuffle(predicted_factors.ravel())#RANDOM
    V_2[V_ind_test_idx] = predicted_factors

    new_test_idx_to_songs()
    
def main(hard,num_clusters):
    start_time = time.time()
    load_data()
    if(hard):
        features_dir = '../featuresnpy/hard/'+str(NUM_FRAMES)+'/'
    else:
        features_dir = '../featuresnpy/soft/'+str(NUM_FRAMES)+'/'
    build_x_y(features_dir,hard,num_clusters)
    mse = regression()
    replace()
    #dump new V's
    data_1 ={
    'U':U,
    'V_old': V,
    'V_new': V_1,
    'test_idx':set(V_ind_test_idx),
    }
    #dump new V's
    data_2 ={
    'U':U,
    'V_old': V,
    'V_new': V_2,
    'test_idx':set(V_ind_test_idx),
    }
    data_3 ={
    'U':U,
    'V_old': V,
    'V_new': V_3,
    'test_idx':set(V_ind_test_idx),
    }
    
    with open(TEST_SONGS_FILE, 'w') as f:
        pickle.dump(test_songs, f, pickle.HIGHEST_PROTOCOL)
    print "data saved to %s" %(TEST_SONGS_FILE)
    with open(FACTORS_FILE_2, 'w') as f:
        pickle.dump(data_2, f, pickle.HIGHEST_PROTOCOL)
    print "data saved to %s" %(FACTORS_FILE_2)
    with open(FACTORS_FILE_1, 'w') as f:
        pickle.dump(data_1, f, pickle.HIGHEST_PROTOCOL)
    print "data saved to %s" %(FACTORS_FILE_1) 
    with open(FACTORS_FILE_3, 'w') as f:
        pickle.dump(data_3, f, pickle.HIGHEST_PROTOCOL)
    print "data saved to %s" %(FACTORS_FILE_3) 
    
    print "generate new V's took %.4f seconds" %(time.time()-start_time)
main(False,700)