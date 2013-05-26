'''
Created on 20 nov. 2012
This parser combines the lyrics and the echonest features
@author: Erik Vandeputte
'''
import echonestparser
import lyricsparser
import cPickle as pickle
import numpy as np

SOURCE_DATA_FILE = "../../msd_dense_subset/mood.txt"
SOURCE_DATA_FILE_2 = "../../msd_dense_subset/mood2.txt"

COMBINED_TARGET_DATA_FILE = "../../msd_dense_subset/mood.pkl"

LYRICS_TARGET_DATA_FILE = "../../msd_dense_subset/mood_lyrics_features_2.pkl"

ECHONEST_TARGET_DATA_FILE = "../../msd_dense_subset/mood_echonest_features_2.pkl"

#parse echonestfeatures
echonestparser.parse_file(SOURCE_DATA_FILE_2, ECHONEST_TARGET_DATA_FILE)
#parse lyricsfeatures
lyricsparser.parse_file(SOURCE_DATA_FILE_2,LYRICS_TARGET_DATA_FILE,1)

#get both features and combine them to a new feature space
with open(ECHONEST_TARGET_DATA_FILE, 'r') as f:
            data = pickle.load(f)
            echonest_features = data['features']
            echonest_tracks = data['tracks']
with open(LYRICS_TARGET_DATA_FILE, 'r') as f:
            data = pickle.load(f)
            labels = data['labels']
            lyrics_features = data['features']
            lyrics_tracks = data['tracks']
combined_tracks = list()
combined_features = np.empty((len(lyrics_tracks),echonest_features.shape[1]+lyrics_features.shape[1]),dtype='float')
combined_labels = list()
for i in range(len(lyrics_tracks)):
    trackid = lyrics_tracks[i]
    if(trackid in echonest_tracks):
            combined_tracks.append(trackid)
            combined_labels.append(labels[i])
            combined_features[i] = np.concatenate((echonest_features[echonest_tracks.index(trackid)],lyrics_features[i]))
data = {
    'features': combined_features,
    'labels': combined_labels,
    'tracks': combined_tracks, #(songindex, songid)
}

with open(COMBINED_TARGET_DATA_FILE, 'w') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
print "data saved to %s" % COMBINED_TARGET_DATA_FILE

