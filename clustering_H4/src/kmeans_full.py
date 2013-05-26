'''
Created on 17 mrt. 2013
My own implementation of the kmeans algorithm
In this implementation the dataset is divided in different batches on HDD
@author: Erik Vandeputte
'''

from scipy import spatial
from scipy import sparse
import cPickle as pickle
import numpy as np
import time 
import os
import random

NUM_FRAMES = 5
NUM_DCT_COEF = 24
MFCCTXTPATH = '../mfccnpy/whitened/'
BATCHES_DIR = '../batches/'
EPS = 0


def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period and retrieves only the mfcc files."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


def kcluster(initial_centroids,batches,k=800,num_iterations=20): 
    
    
    centroids = initial_centroids
   
    loss = 0 #initial loss
    for t in range(num_iterations):
        print 'Iteration %d:%f' %(t,loss)
        last_loss = loss
        loss = 0
        c2 = 0.5* np.sum(np.power(centroids,2),1)
        
        summation = np.zeros((k,centroids.shape[1]))
        counts = np.zeros((k,1))
        
        
        for index,batch in enumerate(batches): #For each Batch, 
            X = np.load(os.path.join(BATCHES_DIR,batch))#open .npy file
            x2 = np.sum(np.power(X,2),1)
            
            temp = np.dot(centroids,X.T)
            x = (temp.T - c2.T).T
            labels = np.argmax(x,axis=0)
            val = np.max(x,axis=0)
            
            loss = loss + np.sum(0.5*x2 - val.T);
            S = sparse.eye(k,k,format='csr')
            S = S[:,labels]
            summation = summation + S.dot(X)
            counts = counts + S.sum(axis=1)
        
        if(last_loss - loss <= EPS and t !=0): #check for convergence
            break
        centroids = summation / np.asarray(counts)
        
        #just zap empty centroids so they don't introduce NaNs everywhere.
        badIndex = np.where(np.asarray(counts) == 0)[0]
        centroids[badIndex] = 0;
        
    return centroids
def make_initial_centroids(num_clusters):
    data = np.empty((num_clusters,NUM_DCT_COEF*NUM_FRAMES))
    files = mylistdir(MFCCTXTPATH)
    for i in range(num_clusters):
        r = random.randint(0,len(files)-1)
        mfccs = np.load(os.path.join(MFCCTXTPATH,files[r])) #open a random .npy file
        startframe = random.randint(0,mfccs.shape[0]-NUM_FRAMES)
        frames = mfccs[startframe:startframe+NUM_FRAMES]
        data[i] = frames[:,0:NUM_DCT_COEF].flatten()
    return data
    
def assign(centroids,X):
    distances = spatial.distance.cdist(X,centroids)
    return np.argmin(distances,axis=1)

def main(num_clusters): #function for training script
    print "starting the algorithm..."
    #load data
    batches = mylistdir(BATCHES_DIR)
    #make initial_centroids
    print "making initial centroids..."
    initial_centroids = make_initial_centroids(num_clusters)
    batches = mylistdir(BATCHES_DIR)
    print "running k-means..."
    start_time = time.time()
    #run K-means
    centroids = kcluster(initial_centroids,batches,num_clusters)
    target_file = '../pklfiles/clusters_'+str(NUM_FRAMES)+'_'+str(num_clusters)+'.pkl'
    with open(target_file, 'w') as f:
            pickle.dump(centroids, f, pickle.HIGHEST_PROTOCOL)
    print "running k-means took %.2f seconds" % (time.time() - start_time)
'''    
#SCRIPT
#load data
initial_centroids = np.load(os.path.join(BATCHES_DIR,'initial_centroids.npy'))
batches = mylistdir(BATCHES_DIR)
total_datapoints = 0
for index,batch in enumerate(batches):
    data = np.load(os.path.join(BATCHES_DIR,batch))#open .npy file
    total_datapoints += data.shape[0]
start_time = time.time()
print "print total number of data points: %s" % str(total_datapoints)

print "running k-means..."
start_time = time.time()
#run K-means
centroids = kcluster(initial_centroids,batches)
print "running k-means took %.2f seconds" % (time.time() - start_time)

with open(TARGET_FILE, 'w') as f:
        pickle.dump(centroids, f, pickle.HIGHEST_PROTOCOL)'''