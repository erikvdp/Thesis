'''
Created on 18 mrt. 2013
preforms whitening for the full dataset
parameters are optained from a MATLAB script
@author: Erik Vandeputte
'''
import numpy as np
import os 


P_FILE = '/Users/Enrico/Documents/MATLAB/P.txt'
M_FILE = '/Users/Enrico/Documents/MATLAB/M.txt'
SOURCE_DIR = '../mfccnpy/original/'
TARGET_DIR = '../mfccnpy/whitened/'

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period and retrieves only the mfcc files."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def load_data():
    global P,M
    #load mean and
    P = np.loadtxt(P_FILE)
    M = np.loadtxt(M_FILE)
    print 'test'
def whitening():
    files = mylistdir(SOURCE_DIR)
    for i,f in enumerate(files):
        if i % 100 == 0:
            print '%d tracks done' %i
        X = np.load(os.path.join(SOURCE_DIR,f))#open .npy file
        W = np.dot(X - M,P) #whiten
        np.save(os.path.join(TARGET_DIR,f),W)
'''
SCRIPT
'''
load_data()
whitening()