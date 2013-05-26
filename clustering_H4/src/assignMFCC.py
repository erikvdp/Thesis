'''
Created on 27 feb. 2013

@author: Erik Vandeputte
Load all the data and assign each MFCC to its nearest center
1) classical hard assignement
2) soft assignement via triangle function: http://ai.stanford.edu/~ang/papers/nipsdlufl10-AnalysisSingleLayerUnsupervisedFeatureLearning.pdf
'''

import cPickle as pickle
from scipy import spatial
from scipy import sparse
import numpy as np
import os
import time

NUM_FRAMES = 100
NUM_DCT_COEF = 12
KMEANS_FILE= '../pklfiles/clusters_'+str(NUM_FRAMES)+'.pkl'
MFCCPATH = '../mfccnpy/'
FEATURE_PATH_HARD = '../featuresnpy/hard/'+str(NUM_FRAMES)+'/'
FEATURE_PATH_SOFT = '../featuresnpy/soft/'+str(NUM_FRAMES)+'/'

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period and retrieves only the mfcc files."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]
    
def hard_assignment():
    start_time = time.time()
    print "calculating features"        
    files = mylistdir(MFCCPATH)
    for i, file in enumerate(files):
        mfccs = np.load(os.path.join(MFCCPATH,file))#open .npy file
        mfccs = mfccs[:,0:NUM_DCT_COEF]
        if(NUM_FRAMES != 1): #ALIGN THE MFCCS SO THAT THEY CORRESPOND TO THE FRAMES (NON-OVERLAPPING)
            num_coef = mfccs.shape[0]
            mfccs = np.resize(mfccs,((num_coef/NUM_FRAMES)+1,NUM_FRAMES*NUM_DCT_COEF))
        features = est.predict(mfccs)
        new_file_name = file[:-4]+'_features'
        np.save(os.path.join(FEATURE_PATH_HARD,new_file_name), features)
        #with open(os.path.join(FEATURE_PATH_HARD,new_file_name), 'w') as f:
        #    pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
        if (i%100 == 0):
            print "%d tracks done" %i
    
    print "assigning datapoints took %.2f seconds" % (time.time() - start_time) # 76s to assign every MFCC of every track


def soft_assignment():
    start_time = time.time()
    centers = est.cluster_centers_ 
    num_clusters = len(centers)
    print "calculating features"        
    files = mylistdir(MFCCPATH)
    for i, file in enumerate(files):
        mfccs = np.load(os.path.join(MFCCPATH,file))#open .npy file
        # ASSIGN MFCCS
        mfccs = mfccs[:,0:NUM_DCT_COEF]
        if(NUM_FRAMES != 1): #ALIGN THE MFCCS SO THAT THEY CORRESPOND TO THE FRAMES (NON-OVERLAPPING)
            num_coef = mfccs.shape[0]
            mfccs = np.resize(mfccs,((num_coef/NUM_FRAMES)+1,NUM_FRAMES*NUM_DCT_COEF))
        #calculate norms
        distances = spatial.distance.cdist(mfccs,centers)
        means = distances.mean(axis=1)
        means = np.repeat(means,num_clusters)
        means = np.reshape(means,(-1,num_clusters))
        features = np.maximum(0,means-distances)
        new_file_name = file[:-4]+'_features'
        np.save(os.path.join(FEATURE_PATH_SOFT,new_file_name), features) #TAKES MORE SIZE
        #with open(os.path.join(FEATURE_PATH_SOFT,new_file_name), 'w') as f:
        #    pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
        if (i%100 == 0):
            print "%d tracks done in %.2f seconds" %(i,(time.time() - start_time))
    
    print "assigning datapoints took %.2f seconds" % (time.time() - start_time)


#load cluster means
with open(KMEANS_FILE, 'r') as f:
    est = pickle.load(f)

hard_assignment()