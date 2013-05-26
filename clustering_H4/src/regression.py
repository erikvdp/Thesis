'''
Created on 17 mrt. 2013

@author: Erik Vandeputte
'''
import cPickle as pickle
import time
import numpy as np
import os

from sklearn import linear_model
from sklearn import cross_validation

INTERACTION_MATIRX_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
SONG_FACTORS_FILE = '../../msd_dense_subset/song_factors.pkl' #more advanced = 200
SONG_7DIGITALID_FILE = "../pklfiles/songs_7digitalids.pkl"

FACTORS_FILE ="../pklfiles/factors"
NUM_FACTORS = 10
NUM_FRAMES = 5

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def load_data():
    global V, songid_ind, ind_songid,song_7digitalid, digitalid_song
    with open(SONG_FACTORS_FILE, 'r') as f:
        data = pickle.load(f)
        V = data['V']
        songid_ind = data['songs_map'] #(songid, songnr)
        ind_songid = data['songs_map_inv']#(songnr, songid)
    with open(SONG_7DIGITALID_FILE, 'r') as f:
        song_7digitalid = pickle.load(f)
        digitalid_song = {v:k for k, v in song_7digitalid.items()}
    with open(INTERACTION_MATIRX_FILE) as f:
        data = pickle.load(f)
        del data
        
def build_x_y_multilevel(features_dir,hard,num_clusters): #hierarchical kmeans
    global X,y
    files = mylistdir(features_dir)
    X = np.empty((len(files),num_clusters*3))
    y = np.empty((len(files),NUM_FACTORS))
    for index,f in enumerate(files):
        digitalid = f[0:f.find('_')]
        songid = digitalid_song[digitalid]
        ind = songid_ind[songid]
        y[index] = V[ind]
        data1 = np.load(os.path.join(features_dir,f))
        data2 = np.load(os.path.join('../featuresnpy/soft/10/',f)) #when working in hierarchical structure
        data3 = np.load(os.path.join('../featuresnpy/soft/20/',f)) #when working in hierarchical structure
        if(hard): #data is a (NUM_SAMPLES,1) matrix with as value its closest centroid
            features1 = np.eye(num_clusters)[data1] #when working in hierarchical structure
            features2 = np.eye(num_clusters)[data2] #when working in hierarchical structure
            features3 = np.eye(num_clusters)[data3] #
            features =np.concatenate((features1.sum(axis = 0).T,features2.sum(axis=0).T,features3.sum(axis=0).T),axis=1) #when working in hierarchical structure
        else:
            features = np.concatenate((data1.sum(axis = 0).T,data2.sum(axis=0).T,data3.sum(axis=0).T),axis=1) #when working in hierarchical structure
        X[index] = features
        
def build_x_y(features_dir,hard,num_clusters): #1level kmeans
    global X,y
    files = mylistdir(features_dir)
    X = np.empty((len(files),num_clusters))
    y = np.empty((len(files),NUM_FACTORS))
    for index,f in enumerate(files):
        digitalid = f[0:f.find('_')]
        songid = digitalid_song[digitalid]
        ind = songid_ind[songid]
        y[index] = V[ind]
        data = np.load(os.path.join(features_dir,f))
        if(hard): #data is a (NUM_SAMPLES,1) matrix with as value its closest centroid
            features = np.eye(num_clusters)[data]
            features = features.sum(axis = 0)
        else:
            features = data.sum(axis = 0)
        X[index] = features
        
def load_x_y():
    global X,y
    print "loading X & y"
    with open('X_Y.pkl', 'r') as f:
        data = pickle.load(f)
        X = data['X']
        y = data['y']
    
def regression():
    global predicted_factors,true_factors
    clf = linear_model.RidgeCV(alphas=np.array([ 0.001,0.01,0.1,1,100,1000,10000,100000,1000000,10000000]))
    X_train, X_validation, y_train, y_validation = cross_validation.train_test_split(X, y, test_size=0.2, random_state=0)
    #test = X.astype(np.int)
    #count = np.bincount(test.flatten())
    #np.random.shuffle(X_train)
    clf.fit(X_train,y_train)
    # The mean square error on the training set
    mse = np.mean((clf.predict(X_train) - y_train) ** 2)
    print ("training MSE: %.4f" %mse)
    # The mean square error on the test set
    y_predictions = clf.predict(X_validation)
    mse = np.mean((y_predictions - y_validation) ** 2)
    var = np.var(y_validation)
    print "(mse/var)=%.4f" %(mse/var)
    mse_per_factor = np.mean((y_predictions - y_validation) ** 2,axis=0)
    print ("test MSE: %f" %mse)
    predicted_factors = clf.predict(X_validation)
    true_factors = y_validation
    print mse_per_factor
    print "alpha:%f" %clf.alpha_
    return mse,mse_per_factor

def main(hard,num_clusters):
    load_data()
    if(hard):
        features_dir = '../featuresnpy/hard/'+str(NUM_FRAMES)+'/'
    else:
        features_dir = '../featuresnpy/soft/'+str(NUM_FRAMES)+'/'
    load_x_y()
    #build_x_y(features_dir, hard, num_clusters)
    mse = regression()
    #dump factors
    data ={
    'y_pred': predicted_factors,
    'y_true': true_factors,
    }
    print "saving results to %s" %(FACTORS_FILE+'_'+str(hard)+'_'+str(num_clusters)+'.pkl')
    with open(FACTORS_FILE+'_'+str(hard)+'_'+str(num_clusters)+'.pkl', 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return mse
main(False,700)
#SCRIPT
#start_time = time.time()
#load_data()
#print "loading data took %.2f seconds" % (time.time() - start_time)
#start_time = time.time()    
#build_x_y('../featuresnpy/soft/10/',False,300)
#print "building features took %.2f seconds" % (time.time() - start_time)    
#start_time = time.time()
#regression()
#print "performing regression took %.2f seconds" % (time.time() - start_time)