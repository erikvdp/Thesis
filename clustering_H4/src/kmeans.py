'''
Created on 2 mrt. 2013
My own implementation of the K-means algorithm
In this implementation the whole dataset is stored in the RAM before running the algorithm
@author: Enrico
'''
from scipy import spatial
from scipy import sparse
import cPickle as pickle
import numpy as np
import time 

NUM_FRAMES = 10
TARGET_FILE = '../pklfiles/clusters_'+str(NUM_FRAMES)+'.pkl'
BATCH_SIZE = 1000
EPS = 0

def kcluster(X,k=500,num_iterations=100): 
    
    x2 = np.sum(np.power(X,2),1) #power to the square and take sum of each row
    
    centroids = X[0:k]
   
    loss = np.sum(x2) #initial loss
    for t in range(num_iterations):
        print 'Iteration %d:%f' %(t,loss)
        last_loss = loss
        loss = 0
        c2 = 0.5* np.sum(np.power(centroids,2),1)
        
        summation = np.zeros((k,centroids.shape[1]))
        counts = np.zeros((k,1))
        
        
        for i in range(0,X.shape[0],BATCH_SIZE): #For each Batch, 
            last_index = min(i+BATCH_SIZE, X.shape[0]);
            m = last_index - i + 1;
            
            temp = np.dot(centroids,X[i:last_index].T)
            x = (temp.T - c2.T).T
            labels = np.argmax(x,axis=0)
            val = np.max(x,axis=0)
            
            loss = loss + np.sum(0.5*x2[i:last_index] - val.T);
            S = sparse.eye(k,k,format='csr')
            S = S[:,labels]
            summation = summation + S.dot(X[i:last_index])
            counts = counts + S.sum(axis=1) 
        
        if(last_loss - loss <= EPS): #check for convergence
            break
        centroids = summation / np.asarray(counts)
        
    return centroids

def assign(centroids,X):
    distances = spatial.distance.cdist(X,centroids)
    return np.argmin(distances,axis=1)

#SCRIPT
#load data
SOURCE_FILE = '../pklfiles/subset_10.npy' #contains random MFCC's extracted with randommfcc.py 
start_time = time.time()
data = np.load(SOURCE_FILE)
print "reading the data took %.2f seconds" % (time.time() - start_time)
print "print total number of data points: %s" % str(data.shape)

print "running k-means..."
start_time = time.time()
#run K-means
centroids = kcluster(data)
print "running k-means took %.2f seconds" % (time.time() - start_time)

with open(TARGET_FILE, 'w') as f:
        pickle.dump(centroids, f, pickle.HIGHEST_PROTOCOL)

print "assigning..."
start_time = time.time()
features = assign(centroids,data)
print "assigning k-means took %.2f seconds" % (time.time() - start_time)