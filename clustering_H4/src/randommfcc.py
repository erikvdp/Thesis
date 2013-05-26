'''
Created on 27 feb. 2013

This file creates a random dataset to preform the training of K-means
total trainingdata size = NUM_SAMPLES * NUM_FRAMES => ~300000
@author: Erik Vandeputte
'''

import random
import time
import numpy as np
import os

NUM_SAMPLES = 1000000 #number of samples to draw
NUM_FRAMES = 1 #number of consecutive frames to extract
NUM_DCT_COEF = 24 #number of DCT coef to take into account
MFCCPATH = '../mfccnpy/original/'
OUTPUTPATH = '../pklfiles/'
TARGET_FILE = 'subset_'+str(NUM_FRAMES)

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]


data = np.empty((NUM_SAMPLES,NUM_DCT_COEF*NUM_FRAMES))

start_time = time.time()
files = mylistdir(MFCCPATH)
for i in range(NUM_SAMPLES):
    r = random.randint(0,len(files)-1)
    mfccs = np.load(os.path.join(MFCCPATH,files[r]))#open .npy file
    #if(mfccs.shape[0] != 2905):
        #print mfccs.shape,files[r]
    startframe = random.randint(0,mfccs.shape[0]-NUM_FRAMES)
    frames = mfccs[startframe:startframe+NUM_FRAMES]
    data[i] = frames[:,0:NUM_DCT_COEF].flatten()
    if(i % (NUM_SAMPLES/100) == 0):
        print '%d percent done' %(i/(NUM_SAMPLES/100))

np.save(os.path.join(OUTPUTPATH,TARGET_FILE), data) #save npy

print "creating the subset took %.2f seconds" % (time.time() - start_time) 
#20s for 30000 mfccs, 140s for 300000 mfccs with NUM_FRAMES = 1
