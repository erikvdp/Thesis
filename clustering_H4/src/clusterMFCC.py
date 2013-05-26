'''
Created on 22 feb. 2013

@author: Erik Vandeputte
this file computes a dictionary of N typical
MFCC vectors over the training set (using K-means class from sk-learn)
http://www.cs.cmu.edu/~chongw/papers/WestonWangWeissBerenzeig2012.pdf
UPDATE: Use MiniBatchKMeans to speed up things
http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf
'''

import numpy as np
import cPickle as pickle
from sklearn.cluster import KMeans, MiniBatchKMeans
import time

NUM_CLUSTERS = 800 #setting to 250 generates a LOT of data
NUM_FRAMES = 100
SOURCE_FILE = '../pklfiles/subset_'+str(NUM_FRAMES)+'.npy' #contains random MFCC's extracted with randommfcc.py 
##SOURCE_FILE = '../pklfiles/subset_small.npy' #contains random MFCC's extracted with randommfcc.py
TARGET_FILE = '../pklfiles/clusters_'+str(NUM_FRAMES)+'.pkl'
NUM_ITERATIONS = 1 #number of iterations for MiniBatch k-means"

start_time = time.time()
data = np.load(SOURCE_FILE)
print "reading the data took %.2f seconds" % (time.time() - start_time)
print "print total number of data points: %s" % str(data.shape)

#print "running k-means..."
#start_time = time.time()
#est = KMeans(n_clusters=NUM_CLUSTERS,init='k-means++',n_jobs=-1)
#est.fit(data)
#print "running k-means took %.2f seconds" % (time.time() - start_time) 
#40s voor 30000 samples, 1623s voor 300000 samples (1% van totale data) voor NUM_FRAMES = 1 & NUM_CLUSTERS = 100
#51s voor 30000 samples, voor NUM_FRAMES = 2
#717s voor 10000 samples (1% van de totale data) voor NUM_FRAMES = 300 & NUM_CLUSTERS = 100

print "running MiniBatch k-means..."
best_average = 0
for i in range(NUM_ITERATIONS):
    start_time = time.time()
    est = MiniBatchKMeans(n_clusters=NUM_CLUSTERS,init='random',max_iter = 1000,batch_size = 100000,reassignment_ratio=0.7)
    est.fit(data)
    predictions = est.predict(data)
    if(np.average(predictions)> best_average):
        best_est = est
        best_average = np.average(predictions)
    print "running iteration %d of MiniBatch k-means took %.2f seconds with average %d" % (i,time.time() - start_time,np.average(predictions)) 
est = best_est
            
with open(TARGET_FILE, 'w') as f:
        pickle.dump(est, f, pickle.HIGHEST_PROTOCOL)

print "assigning datapoints to clusters ..."
start_time = time.time()
predictions = est.predict(data)
counts = np.bincount(predictions)
print "total number of predictions: %d" %predictions.shape[0]
print "optimum points / cluster: %2.f" %(predictions.shape[0]/NUM_CLUSTERS)
print "average + variance  points / cluster: %2.f %2.f" %(np.average(predictions),np.var(predictions))
print np.bincount(predictions)
print "assigning datapoints took %.2f seconds" % (time.time() - start_time) #0.07 voor 30000 samples