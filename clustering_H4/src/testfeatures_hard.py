'''
Created on 6 mrt. 2013
script that searches for a song, based on the dictionary representation, the most identical other song
@author: Erik Vandeputte
'''
import numpy as np
import random as rand
import os

SOURCE_FEATURES_PATH = '../featuresnpy/hard/10'
TRESHOLD = 25
SOURCE_FILE= "144_features.npy"


NUM_ITERATIONS = 10

def mylistdir(directory,exluded_file):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period and retrieves only the mfcc files."""
    filelist = os.listdir(directory)
    return [x for x in filelist if not (x.startswith('.') or x == exluded_file)]

def single_similarity():
    features = np.load(os.path.join(SOURCE_FEATURES_PATH,SOURCE_FILE))#open .npy file
    files = mylistdir(SOURCE_FEATURES_PATH, SOURCE_FILE)
    best_sim_score = 0 
    best_files = []
    best_features = []
    for f in files:
        new_features = np.load(os.path.join(SOURCE_FEATURES_PATH,f))
        if(new_features.shape[0] == features.shape[0]):
            sim_score = sum((new_features == features))
            if (sim_score >= TRESHOLD):
                best_sim_score = sim_score
                best_files.append(f)
                best_features.append(new_features)
    print best_files
    print "number of same clustersassignments: %d/%d)" %(best_sim_score,features.shape[0])
    print features
    print best_features
    
def multiple_similarity():
    best_score = float(0)
    all_files = mylistdir(SOURCE_FEATURES_PATH, "")
    for i in range(NUM_ITERATIONS):
        print "iteration %d" %i
        print "best_score: %.2f" %best_score
        source_file = all_files[rand.randint(0,len(all_files)-1)]
        features = np.load(os.path.join(SOURCE_FEATURES_PATH,source_file))#open .npy file
        files = mylistdir(SOURCE_FEATURES_PATH, source_file)
        for f in files:
            new_features = np.load(os.path.join(SOURCE_FEATURES_PATH,f))
            if(new_features.shape[0] == features.shape[0]):
                sim_score = float(sum((new_features == features)))/float(features.shape[0])
                if (sim_score > best_score):
                    best_score = sim_score
                    best_source = source_file
                    best_source_features = features
                    best_target = f
                    best_target_features = new_features
        print best_source
        print best_target
    print best_source
    print best_target
    print best_source_features
    print best_target_features
    print "best score: %.2f" %(best_score)
    
multiple_similarity()