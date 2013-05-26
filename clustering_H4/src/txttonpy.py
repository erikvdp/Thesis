'''
Created on 25 feb. 2013

@author: Erik Vandeputte
This file reads mfcc coefficients and stores them
to an npy file. This allows for faster processing
'''
import os
import numpy as np
import cPickle as pickle

SONG_7DIGITAL = '../pklfiles/songs_7digitalids.pkl'
MFCCTXTPATH = '../mfcctxt'
MFCCNPYPATH = '../mfccnpy/original'

#read songidsdigitalids dict
#print "load songdigitalids dict"
#with open(SONG_7DIGITAL, 'r') as f:
#    song_digitaldict = pickle.load(f)

#digital_songdict = dict((v,k) for k, v in song_digitaldict.iteritems()) #reverse the mapping

filenames = os.listdir(MFCCTXTPATH)

for index,filename in enumerate(filenames):
    digitalid = filename[0:-4]
    #songid = digital_songdict[digitalid]
    mfcc = np.genfromtxt(os.path.join(MFCCTXTPATH,filename))
    #save in npy format
    np.save(os.path.join(MFCCNPYPATH,digitalid), mfcc)
    if index % 100 == 0:
        print "%d tracks done" %index


