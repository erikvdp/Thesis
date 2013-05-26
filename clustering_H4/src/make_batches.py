'''
Created on 16 mrt. 2013
This script creates the different batches and saves them in the directory /batches
This script also creates the initial centroids
@author: Erik Vandeputte
'''
import os
import numpy as np

OUTPUT_DIR = '../batches/'
SOURCE_DIR = '../mfccnpy/whitened'

BATCH_SIZE = 200000
NUM_FRAMES = 3 #consecutive frames to extract
NUM_DCT_COEF = 24 #number of mfcc coef to take into account

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period and retrieves only the mfcc files."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]
def make_batches():    
    files = mylistdir(SOURCE_DIR)
    
    batch_index = 0
    data = np.empty((BATCH_SIZE,NUM_DCT_COEF*NUM_FRAMES))
    cols = NUM_FRAMES
    data_ptr = 0
    for i,file in enumerate(files):
        mfccs = np.load(os.path.join(SOURCE_DIR,file))#open .npy file
        rows = (mfccs.shape[0]/cols) #+1 => niet wegsmijten maar toevoegen?
        X = np.resize(mfccs,(rows,cols*NUM_DCT_COEF))
        #does it still fit in the batch?
        if (data_ptr + rows < BATCH_SIZE):
            data[data_ptr:data_ptr+rows,:] = X
            data_ptr += rows
        else:
            #divide the X in 2 parts 
            slice_ind = BATCH_SIZE - data_ptr
            data[data_ptr:data_ptr+slice_ind] = X[0:slice_ind]
            #save batch and create a new one
            file = "batch_"+str(batch_index)
            np.save(os.path.join(OUTPUT_DIR,file), data)
            data = np.empty((BATCH_SIZE,NUM_DCT_COEF*NUM_FRAMES))
            batch_index +=1
            data_ptr = 0
            data[data_ptr:rows - slice_ind] = X[slice_ind:]
            data_ptr += rows - slice_ind
        if (i % 1000 == 0):
            print "%d files done" %i
    #save the last batch
    data = np.resize(data,(data_ptr,cols*NUM_DCT_COEF))
    file = "batch_"+str(batch_index)
    np.save(os.path.join(OUTPUT_DIR,file), data)

'''
SCRIPT
'''
make_batches()