'''
Created on 10 mrt. 2013

@author: Erik Vandeputte
'''
import numpy as np
import os

SOURCE_FEATURES_PATH = '../featuresnpy/soft/100'
SOURCE_FILE= "15691_features.npy"

def mylistdir(directory,exluded_file):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period and retrieves only the mfcc files."""
    filelist = os.listdir(directory)
    return [x for x in filelist if not (x.startswith('.') or x == exluded_file)]

def test_soft(features, features_new):
    x = abs(np.subtract(features, new_features))
    return np.sum(x)

features = np.load(os.path.join(SOURCE_FEATURES_PATH,SOURCE_FILE))#open .npy file
files = mylistdir(SOURCE_FEATURES_PATH, SOURCE_FILE)
best_sim_score = 10000000 #some big number
for f in files:
    new_features = np.load(os.path.join(SOURCE_FEATURES_PATH,f))
    if(new_features.shape[0] == features.shape[0]):
        sim_score = test_soft(features, new_features)
        if(sim_score < best_sim_score):
            best_sim_score = sim_score
            best_files = f
            best_features = new_features
print best_files
print "similarity score: %d)" %(-best_sim_score)
print features
print best_features