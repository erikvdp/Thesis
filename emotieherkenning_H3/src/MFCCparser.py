'''
Created on 31 mrt. 2013

@author: Erik Vandeputte
'''

import numpy as np
import cPickle as pickle
import util as ut
import hdf5_getters as GETTERS
import os
import time


SONG_7DIGITALID_FILE = "../../../clusterdata/pklfiles/songs_7digitalids.pkl"
SONG_TRACK_FILE = "../../../msd_dense_subset/dense/songs_tracks.pkl"

SOURCE_DATA_FILE = "../../../msd_dense_subset/mood.txt"
SOURCE_DATA_FILE_2 = "../../../msd_dense_subset/mood2.txt"
MFCC_TARGET_DATA_FILE = "../../../msd_dense_subset/mood_mfcc_features.pkl"
MFCC_TARGET_DATA_FILE_2 = "../../../msd_dense_subset/mood_mfcc_features_2.pkl"

FEATURESDIR = "../../../clusterdata/featuresnpy/soft/10"

NUM_FEATURES = 1200

def load_data():
    global track_song, song_digitalid, files
    with open(SONG_7DIGITALID_FILE,'r') as f:
        song_digitalid = pickle.load(f)
    with open(SONG_TRACK_FILE,'r') as f:
        data = pickle.load(f)
        track_song = {v:k for k, v in data.items()}
    filelist = os.listdir(FEATURESDIR)
    files = [x for x in filelist
            if not (x.startswith('.'))]

def parse_file(file, targetfile):
    start_time = time.time()
    with open(file, 'r') as f:
        global labels, features
        tracks = list()
        labels = list()
        for line in f:
               track, label = line.strip().split(' ')
               tracks.append(track)
               labels.append(int(label))
        labels = np.array(labels)
        features = np.empty((len(tracks),NUM_FEATURES),dtype='float')#hardcode number of features otherwise problem with scaling
        track_info = np.empty((len(tracks)),dtype='object')
        not_found = 0
        for i,track in enumerate(tracks):
                #print track +' - ' +ut.get_track_info(track)
                #get song
                songid = track_song[track]
                #get id
                digitalid= song_digitalid[songid]
                #open the features
                f = digitalid+"_features.npy"
                if(f in files):
                    data = np.load(os.path.join(FEATURESDIR,f))
                    features[i] = data.sum(axis=0) #samenvatting genereren
                else:
                    not_found +=1
        #remove not found MFCC's
        features = features[~np.all(features == 0, axis=1)]
        labels = labels[~np.all(features == 0, axis=1)]
        #save data
        data = {
        'features': features,
        'labels': labels,
        }
    with open(MFCC_TARGET_DATA_FILE, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    print "data saved to %s" % MFCC_TARGET_DATA_FILE
    print "parsing took %.2f seconds" %(time.time() - start_time)
    print "mfccs not found %d" %not_found
        
def main():
    load_data()
    parse_file(SOURCE_DATA_FILE_2,MFCC_TARGET_DATA_FILE_2)
    
main()