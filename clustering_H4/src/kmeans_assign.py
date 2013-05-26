'''
Created on 13 mrt. 2013
This script assigns the frames of each song to the clusters
clustering is based on kmeans.py
@author: Erik Vandeputte
'''

import cPickle as pickle
from scipy import spatial
from scipy import sparse
import numpy as np
import os
import sys
import time

NUM_FRAMES = 5
NUM_DCT_COEF = 24
KMEANS_FILE= '../pklfiles/clusters_'+str(NUM_FRAMES)+'.pkl'
MFCCPATH = '../mfccnpy/whitened'
FEATURE_PATH_HARD = '../featuresnpy/hard/'+str(NUM_FRAMES)+'/'
FEATURE_PATH_SOFT = '../featuresnpy/soft/'+str(NUM_FRAMES)+'/'

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period and retrieves only the mfcc files."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def assign(centroids,X):
    distances = spatial.distance.cdist(X,centroids)
    return np.argmin(distances,axis=1)


def hard_assignment():
    start_time = time.time()
    print "calculating features"        
    files = mylistdir(MFCCPATH)
    for i, file in enumerate(files):
        mfccs = np.load(os.path.join(MFCCPATH,file))#open .npy file
        num_coef = mfccs.shape[0]
        mfccs = np.resize(mfccs,((num_coef/NUM_FRAMES)+1,NUM_FRAMES*NUM_DCT_COEF))
        features = assign(centroids, mfccs)
        new_file_name = file[:-4]+'_features'
        np.save(os.path.join(FEATURE_PATH_HARD,new_file_name), features)
        if (i%100 == 0):
            print "%d tracks done" %i
    
    print "assigning datapoints took %.2f seconds" % (time.time() - start_time)


def soft_assignment():
    start_time = time.time()
    num_clusters = centroids.shape[0]
    print "calculating features"        
    files = mylistdir(MFCCPATH)
    for i, file in enumerate(files):
        mfccs = np.load(os.path.join(MFCCPATH,file))#open .npy file
        # ASSIGN MFCCS
        num_coef = mfccs.shape[0]
        mfccs = np.resize(mfccs,((num_coef/NUM_FRAMES)+1,NUM_FRAMES*NUM_DCT_COEF))
        new_file_name = file[:-4]+'_features'
        #calculate norms
        distances = spatial.distance.cdist(mfccs,centroids)
        means = distances.mean(axis=1)
        means = np.repeat(means,num_clusters)
        means = np.reshape(means,(-1,num_clusters))
        features = np.maximum(0,means-distances)
        np.save(os.path.join(FEATURE_PATH_SOFT,new_file_name), features) #TAKES MORE SIZE
        if (i%100 == 0):
            print "%d tracks done in %.2f seconds" %(i,(time.time() - start_time))
    
    print "assigning datapoints took %.2f seconds" % (time.time() - start_time)

def shrinkage_assignment(): #TODO: onvolledig
    start_time = time.time()
    num_clusters = centroids.shape[0]
    print "calculating features"
    files = mylistdir(MFCCPATH)
    for i, file in enumerate(files):
        mfccs = np.load(os.path.join(MFCCPATH,file))#open .npy file
        num_coef = mfccs.shape[0]
        mfccs = np.resize(mfccs,((num_coef/NUM_FRAMES)+1,NUM_FRAMES*NUM_DCT_COEF))
        new_file_name = file[:-4]+'_features'
        #calculate norms
        distances= spatial.distance.cdist(mfccs,centroids)   
    
def main(hard,num_clusters):
    global centroids
    kmeans_file = '../pklfiles/clusters_'+str(NUM_FRAMES)+'_'+str(num_clusters)+'.pkl'
    with open(kmeans_file, 'r') as f:
        centroids = pickle.load(f)
    if(hard):
        hard_assignment()
    else:
        soft_assignment()
#main(False,700)