'''
Created on 28 nov. 2012

@author: Erik Vandeputte
'''

from pyechonest import song
from pyechonest import track
from pyechonest import config
from sklearn import preprocessing
import cPickle as pickle
import numpy as np
import urllib2
import json
import sys
CLASSIFIER_FILE = "./classifier" #dump of the classifier that can be used in mood_detection_application

#set API KEY
config.ECHO_NEST_API_KEY="0F5OF2UCBE4BBF7JI"
LAST_FM_API_KEY = "23d4d080ab66300840b2f6cc49151fbb"


#set NUMBER OF TRACKS
NUM_RECENT_TRACKS = 3
#set USERNAME
#USERNAME = 'perikvdp'

def load_data():
    global classifier
    #load classifier
    with open(CLASSIFIER_FILE, 'r') as f:
            data = pickle.load(f)
            classifier = data['classifier']
def load_last_fm(username):
    tracks = list()
    print "predicting mood for %s" %username
    url = "http://ws.audioscrobbler.com/2.0/?method=user.getrecenttracks&user="+username+"&api_key="+LAST_FM_API_KEY+"&format=json&limit="+str(NUM_RECENT_TRACKS)
    data = urllib2.urlopen(url)
    j = json.load(data)
    for track in j['recenttracks']['track']:
        artist = track['artist']['#text']
        title = track['name']
        tracks.append((artist,title))
    return tracks
    
def fetch_features(tracks):
    features = np.zeros((len(tracks),125))
    found_tracks = list()
    for k,track in enumerate(tracks):
        results = song.search(artist=track[0], title=track[1])
        if (len(results) == 0):
            continue
        s = results[0]
        found_tracks.append(track)
        print 'fetching features for track %d: %s - %s' %(k,track[0],track[1])
        
        energy = s.audio_summary['energy']
        avg_loudness = s.audio_summary['loudness']
        tempo =s.audio_summary['tempo']
        
        url = s.audio_summary['analysis_url']
        try:
            data = urllib2.urlopen(url)
            j = json.load(data)
            #fetch features
            segments =j['segments']
            num_segments = len(segments)
            timbres = np.zeros((12,num_segments))
            pitches = np.zeros((12,num_segments))
            start_segments = np.zeros((num_segments))
            loudness_max = np.zeros((num_segments))
            for i in range(num_segments):
                timbre = segments[i]['timbre'] 
                timbres[:,i] = np.array(timbre).T
                pitch = segments[i]['pitches']
                pitches[:,i] = np.array(pitch).T
                start_segments[i] = segments[i]['start']
                loudness_max[i] = segments[i]['loudness_max']
            timbres = timbres.T
            pitches = pitches.T
            timbre = (timbres.mean(axis=0),timbres.var(axis=0), np.median(timbres,axis=0), np.min(timbres,axis=0), np.max(timbres,axis=0))
            pitch = (pitches.mean(axis=0), pitches.var(axis=0), np.median(pitches,axis=0), np.min(pitches,axis=0), np.max(pitches,axis=0))
            idx = np.where((start_segments > j['track']['end_of_fade_in']) & (start_segments < j['track']['start_of_fade_out']))
            loudness =  (avg_loudness, np.var(loudness_max[idx]), abs(max(loudness_max[idx])  - abs(min(loudness_max[idx]))))
            features[k] =  np.concatenate((timbre[0], timbre[1], timbre[2], timbre[3], timbre[4],pitches[0], pitches[1],pitches[2], pitches[3],pitches[4],np.array([tempo]), np.array(loudness),np.array([energy])))
        except:
            print "could not find song"
    #remove zero rows
    idx_nonzero  = np.unique(features.nonzero()[0])
    features = features[idx_nonzero,:]
    return (found_tracks,features)

def classify(tracks,features):
    print  
    results = classifier.predict(features)
    for i,result in enumerate(results):
        if (int(result) ==0 ):
            print '%s - %s classified as: happy' %(tracks[i][0],tracks[i][1])
        else:
            print '%s - %s classified as: sad' %(tracks[i][0],tracks[i][1])
    if len(sys.argv[1:]) == 1:
        if (sum(results) > float(len(results))*0.5):
            print '%s ,Are you currently in a sad mood? : (' %sys.argv[1]
        else:
            print '%s , Are you currently in a happy mood? : )' %sys.argv[1]
        
def usage():
    print 'Usage: python mood_detection_application.py <last.fm username> OR <artist> <title>'
    sys.exit(-1)

def main():
    tracks = list()
    args = sys.argv[1:]
    load_data()
    if len(args) == 1:
        tracks = load_last_fm(args[0])
    elif len(args) == 2:
        track = args[0],args[1]
        tracks.append(track)
    else:
        usage()
    test_tracks, test_features = fetch_features(tracks)
    classify(test_tracks, test_features)

if __name__ == "__main__":
    main()