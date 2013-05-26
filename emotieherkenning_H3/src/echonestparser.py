'''
Created on 2 nov. 2012

@author: Erik Vandeputte
this file parses the training data and gathers the features
'''
import numpy as np
import util as ut
import cPickle as pickle
from sklearn import preprocessing
import hdf5_getters as GETTERS

SOURCE_DATA_FILE = "../../msd_dense_subset/mood.txt"
SOURCE_DATA_FILE_2 = "../../msd_dense_subset/mood2.txt"
E_TARGET_DATA_FILE = "../../msd_dense_subset/mood_echonest_features.pkl"
E_TARGET_DATA_FILE_2 = "../../msd_dense_subset/mood_echonest_features_2.pkl"

NUM_FEATURES = 125

def parse_file(file, targetfile):
    print "print parsing %s" %file
    with open(file, 'r') as f:
        global tracks, labels, features
        tracks = list()
        labels = list()
        for line in f:
            track, label = line.strip().split(' ')
            tracks.append(track)
            labels.append(int(label))
        labels = np.array(labels)
        features = np.empty((len(tracks),NUM_FEATURES),dtype='float')#hardcode number of features otherwise problem with scaling
        track_info = np.empty((len(tracks)),dtype='object')
        for i,track in enumerate(tracks):
            #print track +' - ' +ut.get_track_info(track)
            if(i%100 ==0):
                print "processing track:%s\t%s" %(str(i),str(track))
            if(file == SOURCE_DATA_FILE): #fetch h5 file from small dataset
                h5 = GETTERS.open_h5_file_read("../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5")    #fetch h5 file to allow faster preprocessing
            else:
                h5 = GETTERS.open_h5_file_read("../../msd_dense_subset/mood/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5")    #fetch h5 file to allow faster preprocessing
            track_info[i] = ut.get_track_info(track,h5)
            timbre = ut.get_timbre(track,h5) #returns a tuple with 5 elements (12*5 = 60)
            tempo = ut.get_tempo_feature(track,h5) #(1)
            loudness = ut.get_loudness(track,h5)#returns a tuple with 3 elements (3)
            energy = ut.get_energy_feature(track) #(1)
            pitches = ut.get_pitches(track, h5) #(12)
            features[i] =  np.concatenate((timbre[0], timbre[1], timbre[2], timbre[3], timbre[4],pitches[0], pitches[1],pitches[2], pitches[3],pitches[4],np.array([tempo]),np.array(loudness),np.array([energy])))
            h5.close()
        print "done parsing"
        print "saving data"
        
    data = {
        'features': features,
        'labels': labels,
        'tracks': tracks, #(songindex, songid)
        'track_titles': track_info
    }
    
    with open(targetfile, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    print "data saved to %s" % targetfile

#SCRIPT
#parse_file(SOURCE_DATA_FILE_2,E_TARGET_DATA_FILE_2)
#parse_file(SOURCE_DATA_FILE,E_TARGET_DATA_FILE)