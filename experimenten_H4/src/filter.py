'''
Created on 27 apr. 2013
Deze klasse filtert uit de dataset de nummers waarvoor geen MFCC's gegenereerd konden worden

@author: Erik Vandeputte
'''
import os
import cPickle as pickle


SONG_7DIGITALID_FILE = "./pklfiles/songs_7digitalids.pkl"
FILTER_FILE = './pklfiles/absent_MFCC.pkl'
DATASET_FILE = 'data_dense_subset.txt'
DATASET_2_FILE = 'data_dense_subset_filtered.txt'

MFCCDIR = '../clusterdata/featuresnpy/soft/5/'

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period and retrieves only the mfcc files."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

with open(SONG_7DIGITALID_FILE, 'r') as f:
        song_7digitalid = pickle.load(f)
        digitalid_song = {v:k for k, v in song_7digitalid.items()}



#load MFCC songs (should be about 9950)
MFCC_songids = set()
MFCC_digitalids = mylistdir(MFCCDIR)
for MFCC_digitalid in MFCC_digitalids:
    MFCC_digitalid = MFCC_digitalid[0:MFCC_digitalid.index('_')]
    MFCC_songids.add(digitalid_song[MFCC_digitalid])

#load all songs (should be 10000)
all_songids = set()
with open(DATASET_FILE, 'r') as f:
        for line in f:
            user, song, _ = line.strip().split('\t')
            if song not in all_songids:
                all_songids.add(song)

#intersect
absent_MFCC_ids = all_songids.difference(MFCC_songids)

#write songs who don't have MFCC features to file
print "pickle data"
with open(FILTER_FILE, 'w') as f:
    pickle.dump(absent_MFCC_ids, f, pickle.HIGHEST_PROTOCOL)
    
#write new dataset without the absent_MFCC_ids
with open(DATASET_FILE, 'r') as f,open(DATASET_2_FILE,'w') as f2:
    for line in f:
            user, song, _ = line.strip().split('\t')
            if song not in absent_MFCC_ids:
                f2.write(line)


