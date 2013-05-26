'''
Created on 8 nov. 2012

@author: Erik Vandeputte
'''
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
import cPickle as pickle
from fwrap.fparser.Fortran2003 import Else_If_Stmt

LYRICS_FILE = "../../msd_dense_subset/mxm/mxm_dataset.txt" #contains all lyrics info
SENTIMENT_WORDS = '../../msd_dense_subset/sentiment_words.pkl' 

SOURCE_DATA_FILE = "../../msd_dense_subset/mood.txt"
E_TARGET_DATA_FILE = "../../msd_dense_subset/mood_lyrics_features.pkl"

SOURCE_DATA_FILE_2 = "../../msd_dense_subset/mood2.txt"
E_TARGET_DATA_FILE_2 = "../../msd_dense_subset/mood_lyrics_features2.pkl"

FEATURE_SELECTED_WORDS = '../../msd_dense_subset/feature_selected_words.pkl' #contains the indices of the words obtained from mutual information

def load_sentiment_words():
    with open(SENTIMENT_WORDS, 'r') as f:
        data = pickle.load(f)
        return data['indices']
def parse_file(file, targetfile, option):
    if option == 1:
        sentiment_words_idx = load_sentiment_words()
    else:
        sentiment_words_idx = range(5000)
    print "suitable sentiment words %d" %(len(sentiment_words_idx))
    print "print parsing %s" %file
    with open(file, 'r') as moodfile:
            tracks = list()
            labels = list()
            for line in moodfile:
                track, label = line.strip().split(' ')
                tracks.append(track)
                labels.append(int(label))
    labels = np.array(labels)
    lyrics = np.empty((len(tracks),5000),dtype='float')
    with open(LYRICS_FILE, 'r') as lyricsfile:
        for line in lyricsfile:
            if line.startswith( '#' ):
                continue
            if line.startswith( '%' ): #list of topwords
                popular_words = line.strip().split(',')
                popular_words[0] = "i" #remove the % sign
                popular_words = np.array(popular_words)
                popular_words = popular_words[sentiment_words_idx] #only take in account the sentiment_words
            else:    #normal line, contains track_id, mxm track id, then word count for each of the top words, comma-separated ,word count is in sparse format -> ...,<word idx>:<cnt>,...
                info = line.strip().split(',')
                trackid = info[0]
                if trackid in tracks: #we found a lyric for this track
                    word_array = np.zeros((5000), dtype='float')
                    words = info[2:]
                    for word in words:
                            index, freq = word.split(':')
                            if int(index)-1 in sentiment_words_idx: #is it a sentiment word? #<word idx> starts at 1 (not zero!)
                                word_array[int(index)-1] = float(freq)
                    lyrics[tracks.index(trackid,)] = word_array
    track_idx = np.unique(lyrics.nonzero()[0])
    lyrics  = lyrics[track_idx,:] #reshape lyrics to have only nonzero rows
    lyrics  = lyrics[:,sentiment_words_idx] #reshape lyrics to have only nonzero colums
    labels = labels[track_idx] #only store the labels of the tracks that have lyrics
    tracks = np.array(tracks)
    tracks = tracks[track_idx] #only store the name of the tracks that have lyrics and sentiment_words
    print "number of tracks:%d" %len(tracks)

    
    data = {
        'words': popular_words,
        'features': lyrics,
        'labels': labels,
        'tracks': tracks, #(songindex, songid)
    }
    
    with open(targetfile, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    print "data saved to %s" % targetfile

#SCRIPT 
#parse_file(SOURCE_DATA_FILE,E_TARGET_DATA_FILE,1)
#parse_file(SOURCE_DATA_FILE_2,E_TARGET_DATA_FILE_2,1)