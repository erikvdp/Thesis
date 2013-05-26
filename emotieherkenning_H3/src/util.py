'''
Created on 1 nov. 2012

@author: Erik Vandeputte
'''
import json
import time
import random
import cPickle as pickle
import hdf5_getters as GETTERS
import numpy as np
from pyechonest import song
from pyechonest import config
from sklearn import preprocessing


TRACKS_WITH_TAGS ="../../../msd_dense_subset/lastfm/tracks_with_tags.txt"
TAGFILE = "../../../msd_dense_subset/tags.txt" #matches the subset with the last.fm subset tags
TAGFILE2 = "../../../msd_dense_subset/tags2.txt" #matches the MSD with the last.fm subset tags
TRACKS = "../../../msd_dense_subset/tracks2.txt"
TRACKS_INFO ="../../../msd_dense_subset/tracksinfo.txt" #info about the track (artist, title, label)
MOODFILE = "../../../msd_dense_subset/mood.txt" #mood tracks for subset
MOODFILE2 = "../../../msd_dense_subset/mood2.txt" #mood tracks for MSD dataset
TRAININGFILE = "../../../msd_dense_subset/mood_training.txt"
VALIDATIONFILE = "../../../msd_dense_subset/mood_test.txt"
CROSSFOLDFILE = "../../../msd_dense_subset/mood_crossfold"

ENERGYFILE = "../../msd_dense_subset/energy.pkl" #contains the available energies of tracks who appear in dataset 1 and dataset 2

#set API KEY
config.ECHO_NEST_API_KEY="0F5OF2UCBE4BBF7JI"


def get_tags():
    #get for all 10000 songs the corresponding tags
    songs_tracks = pickle.load(open ("../../msd_dense_subset/dense/songs_tracks.pkl",'r'));
    with open(TAGFILE, 'w') as f1:
        for song in songs_tracks:
            track = str(songs_tracks[song])
            # build path
            path = "../../msd_dense_subset/lastfm/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".json"
            try:
                with open(path, 'r') as f2:
                    entry = json.load(f2)
                    tags = entry["tags"]
                    names = list(x[0] for x in tags)
                    f1.write(track + '|' + ' '.join(names) + '\n')
            except IOError as e:
                print 'Could not find file'
            except UnicodeEncodeError as e:
                print 'Could not print characters'


def get_tags_2():             
#get all trackids witch contain happy sad-tags
    with open(TAGFILE2, 'w') as f1, open(TRACKS, 'w') as f3:
            with open(TRACKS_WITH_TAGS,'r') as f:
                for line in f:
                    track = line.strip() #remove \n
                    # build path
                    path = "../../msd_dense_subset/lastfm/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".json"
                    try:
                        with open(path, 'r') as f2:
                            entry = json.load(f2)
                            tags = entry["tags"]
                        names = list(x[0] for x in tags)
                        if "happy" in names or "sad" in names:
                            try:
                                f1.write(track + '|' + ' '.join(names) + '\n')
                                f3.write(track + "\n")
                            except UnicodeEncodeError as e:
                                print 'Could not print tags for %s' %str(track)
                    except IOError as e:
                        print 'Could not find file for %s' %str(track)
                    
                            
def write_mood(num_classes):
    #this method fetches labels for dataset 1
    moodfile =  open(MOODFILE, 'w')
    trackinfo = open(TRACKS_INFO, 'w')
    happy = 0
    sad = 0
    anger = 0
    with open(TAGFILE, 'r') as tagfile:
        for line in tagfile:
            data = line.strip().split('|') #song,track, (tags)
            track = data[0]
            tags = data[1]
            if("happy" in tags and "sad" in tags):
                continue
            if(tags.count("happy") > 1):
                moodfile.write(track + ' 0' + '\n');
                #trackinfo.write(get_track_info(track) + '- 0' + '\n')
                happy = happy +1
            elif(tags.count("sad") > 1):
                moodfile.write(track + ' 1'+ '\n')
                #trackinfo.write(get_track_info(track) + '- 1' + '\n')
                sad = sad +1
            elif(num_classes > 2 and tags.count("anger") > 0):
                anger = anger +1;
                moodfile.write(track + ' 2' +'\n')
                trackinfo.write(get_track_info(track) + '- 2' + '\n')
                
    moodfile.close()
    trackinfo.close()
    print "total songs: %d" %(happy+sad+anger)
    print "total songs that will be taken into account: %d" %(happy+sad+anger)
    print "number of happy songs: %d" %happy
    print "number of sad songs: %d" %sad
    print "number of anger songs: %d" %anger

def write_mood_2():
    #this method fetches the labels for dataset 2
    moodfile =  open(MOODFILE2, 'w')
    trackinfo = open(TRACKS_INFO, 'w')
    happy = 0
    sad = 0
    with open(TAGFILE2, 'r') as tagfile:
        for line in tagfile:
            data = line.strip().split('|') #song,track, (tags)
            track = data[0]
            tags = data[1]
            if("happy" in tags and "sad" in tags):
                continue
            if(tags.count("happy") > 1):
                moodfile.write(track + ' 0' + '\n');
                happy = happy +1
                trackinfo.write(get_track_info(track) + '- 0' + '\n')
            elif(tags.count("sad") > 1):
                moodfile.write(track + ' 1'+ '\n')
                sad = sad +1
                trackinfo.write(get_track_info(track) + '- 1' + '\n')
    moodfile.close()
    trackinfo.close()
    print "total songs: %d" %(happy+sad)
    print "total songs that will be taken into account: %d" %(happy+sad)
    print "number of happy songs: %d" %happy
    print "number of sad songs: %d" %sad

def get_key_feature(track, h5=None):
    #return
    #0: get key of the track
    #1:get mode (minor = 0, major = 1close = (h5== None)
    close = (h5 == None)
    if h5 == None:
        # build path
        path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)
    mode = GETTERS.get_mode(h5)
    key = GETTERS.get_key(h5)
    confidence_mode = GETTERS.get_mode_confidence(h5)
    confidence_key = GETTERS.get_key_confidence(h5)
    if close:
        h5.close()
    return (key,mode)
    #string_key = ('c', 'c-sharp', 'd', 'e-flat', 'e', 'f', 'f-sharp', 'g', 'a-flat', 'a', 'b-flat', 'b')
    #string_mode = ('minor', 'major')
    #return (string_key[key],string_mode[mode])

def get_timbre(track,h5=None):
    #returns
    #0 mean of all the timbre components in the song
    #1 variances of all timbre components in the song
    #2 median of all timbre components in the song
    #3 min component of all timbre components in the song
    #4 max component of all timbre components in the song
    close = (h5== None)
    if h5==None:
        path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)    
    bttimbre = GETTERS.get_segments_timbre(h5)
    #bttimbre = preprocessing.scale(bttimbre) #todo: ask if it's better to normalize here?
    if close:
        h5.close()
    if (bttimbre != None):
        return (bttimbre.mean(axis=0),bttimbre.var(axis=0), np.median(bttimbre,axis=0), np.min(bttimbre,axis=0), np.max(bttimbre,axis=0))
    else :
        return None

def get_tempo_feature(track,h5=None):
    #get tempo of the song
    close = (h5== None)
    if h5==None:
        path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)
    tempo = GETTERS.get_tempo(h5)
    if close:
        h5.close()
    if (tempo != None):
        return float(tempo)
    else :
        return None
def get_pitches(track,h5=None):
    close = (h5== None)
    if h5==None:
        path = "../../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)
    pitches = GETTERS.get_segments_pitches(h5)
    if close:
        h5.close()
    return (pitches.mean(axis=0), pitches.var(axis=0), np.median(pitches,axis=0), np.min(pitches,axis=0), np.max(pitches,axis=0))
def fetch_energy_feature(track,h5=None):
    #get energy of the track => uses pyechonest library
    close = (h5== None)
    if h5==None:     
        path = "../../msd_dense_subset/mood/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)
    songid = GETTERS.get_song_id(h5)
    if close:
        h5.close()
    results = song.profile(songid)
    if len(results) == 0:
        return 0
    track = results[0]
    energy = track.audio_summary['energy']
    if (energy != None):
        return float(energy)
    else :
        return 0
def get_energy_feature(track):
    with open(ENERGYFILE, 'r') as f:
        data = pickle.load(f)
        return data[track]
def get_loudness(track,h5=None):
    #returns
    #0: the average loudness of the track
    #1: the variance of the loudness over the track
    #2: the difference between the highest and lowest dB value between the end of fade in and start of fade out
    close = (h5== None)
    if h5 == None:
        path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)
    loudness_avg = GETTERS.get_loudness(h5)
    loudnesses_interval = GETTERS.get_segments_loudness_max(h5)
    start_segments = GETTERS.get_segments_start(h5)
    start_fade_out = GETTERS.get_start_of_fade_out(h5)
    end_fade_in = GETTERS.get_end_of_fade_in(h5)
    idx = np.where((start_segments > end_fade_in) & (start_segments < start_fade_out))
    if close:
        h5.close()
    #return (loudness_avg, np.var(loudnesses_interval[idx]), min(loudnesses_interval[idx]),  max(loudnesses_interval[idx]))
    return (loudness_avg, np.var(loudnesses_interval[idx]), abs(max(loudnesses_interval[idx])  - abs(min(loudnesses_interval[idx]))))

def get_time_signature(track,h5=None):
    close = (h5== None)
    if h5==None:
        path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)
    time_signature = GETTERS.get_time_signature(h5)
    if close:
        h5.close()
    return time_signature
    
def get_track_info(track,h5=None):
    #get song and artist of the track
    close = (h5== None)
    if h5==None:
        path = "../../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)
    artist = GETTERS.get_artist_name(h5)
    title = GETTERS.get_title(h5)
    if close:
        h5.close()
    return str(artist) + '-' + str(title)
    
#try to estimate the mode of a track => http://en.wikipedia.org/wiki/Mode_(music)
#returns an int corresponding to the 7 musical modes
#DOESN'T WORK DUE TO BAD KEY ESTIMATION
#SOLUTION = return the 7 most important pitches
def get_mode_advanced_feature(track,h5=None):
    modern_modes = np.array([[1,0,1,0,1,1,0,1,0,1,0,1],
                             [1,0,1,1,0,1,0,1,0,1,1,0],
                             [1,1,0,1,0,1,0,1,1,0,1,0],
                             [1,0,1,0,1,0,1,1,0,1,0,1],
                             [1,0,1,0,1,1,0,1,0,1,1,0],
                             [1,0,1,1,0,1,0,1,1,0,1,0],
                             [1,1,0,1,0,1,1,0,1,0,1,0]])
    close = (h5== None)
    if h5 == None:
        path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
        h5 = GETTERS.open_h5_file_read(path)
    pitches = GETTERS.get_segments_pitches(h5)
    maxpitches = pitches.argmax(axis=1)
    important_pitches = np.bincount(maxpitches).argsort()[-7:] #get the 7 most important pitches
    mode_advanced = np.zeros(12)
    mode_advanced[important_pitches] = 1;
    #key = GETTERS.get_key(h5)
    #mode_advanced = np.roll(mode_advanced,-key)
    #max = 11
    #mode_advanced = 0
    #for i in range(7):
    #    if(sum(mode_advanced != modern_modes[i,:])< max):
    #        max = sum(mode_advanced != modern_modes[i,:])
    #        mode_advanced = i
    if close:
        h5.close()
    return mode_advanced
        
def test_mode():
    results = dict()
    normalizedresults = dict()
    results["happy_minor"] = 0
    results["happy_major"] = 0
    results["sad_minor"] = 0
    results["sad_major"] = 0
    with open(MOODFILE, 'r') as moodfile:
        for line in moodfile:
            track, mood = line.strip().split(' ')
            mode = get_key_feature(track)[1]
            if(mode == 0 and int(mood) == -1 ):
                results["sad_minor"] = results["sad_minor"] + 1
            if(mode == 0 and int(mood) == 1 ):
                results["happy_minor"] = results["happy_minor"] + 1
            if(mode == 1 and int(mood) == -1 ):
                results["sad_major"] = results["sad_major"] + 1
            if(mode == 1 and int(mood) == 1 ):
                results["happy_major"] = results["happy_major"] + 1
    total_songs = map(int, results.values())
    print "total tracks: " + str(sum(total_songs))
    print results
    normalizedresults["happy_minor"] = float(results["happy_minor"])/(results["happy_minor"]+results["happy_major"])
    normalizedresults["happy_major"] = float(results["happy_major"])/(results["happy_minor"]+results["happy_major"])
    normalizedresults["sad_minor"] = float(results["sad_minor"])/(results["sad_minor"]+results["sad_major"])
    normalizedresults["sad_major"] = float(results["sad_major"])/(results["sad_minor"]+results["sad_major"])
    print normalizedresults
def test_tempo():
    tempo_pos = list()
    tempo_neg = list()
    with open(MOODFILE, 'r') as moodfile:
        for line in moodfile:
            track, mood = line.strip().split(' ')
            tempo = get_tempo_feature(track)
            if(int(mood) == -1):
                tempo_pos.append(tempo)
            else:
                tempo_neg.append(tempo)
    print "mean tempo for happy songs: %f BPM" % (sum(tempo_pos)/len(tempo_pos))
    print "mean tempo for sad songs: %f BPM" % (sum(tempo_neg)/len(tempo_neg))
def test_loudness():
    loudness_pos = list()
    loudness_neg = list()
    loudness_dif_pos = list()
    loudness_dif_neg = list()
    with open(MOODFILE, 'r') as moodfile:
        for line in moodfile:
            track, mood = line.strip().split(' ')
            avg, var, loudness_dif = get_loudness(track)
            if(int(mood) == 1):
                loudness_dif_pos.append(loudness_dif)
                loudness_pos.append(avg)
            else:
                loudness_dif_neg.append(loudness_dif)
                loudness_neg.append(avg)
    print "mean loudness + max loudness difference for happy songs: %f dB %f dB" % (np.mean(loudness_pos),sum(loudness_dif_pos)/len(loudness_dif_pos))
    print "mean loudness + max loudness difference for sad songs: %f dB %f dB" % (np.mean(loudness_neg),sum(loudness_dif_neg)/len(loudness_dif_neg))
def test_energy():
    energy_pos = list()
    energy_neg = list()
    energy_anger = list()
    energies = dict()
    i= 1;
    with open(MOODFILE2, 'r') as moodfile:
        for line in moodfile:
            track, mood = line.strip().split(' ')
            print "fetching track %d: %s" %(i,track)
            if (i%50 == 0): #limit server load due to API restrictions
                print "%d tracks done" %i
                print "sleeping for 2 minutes"
                time.sleep(120)
                print "fetch new data!"
            succes = False;
            while(succes == False):
                try:
                    energy = fetch_energy_feature(track);
                    succes = True
                except Exception as e:
                    print 'Kicked out of the API, going to sleep for 2 minutes'
                    time.sleep(120)
            if(int(mood) == -1):
                energy_neg.append(energy)
            if(int(mood) == 1):
                energy_pos.append(energy)
            if(int(mood) == -2):
                energy_anger.append(energy)  
            energies[track] = energy
            i = i+1
    print "mean energy for happy songs: %.2f" %(np.mean(energy_pos))
    print "mean energy for sad songs: %.2f" %(np.mean(energy_neg))
    print "mean energy for anger songs: %.2f" %(np.mean(energy_anger))
    print "saving data"
    
    with open(ENERGYFILE, 'w') as f:
        pickle.dump(energies, f, pickle.HIGHEST_PROTOCOL)
        
    print "data saved to %s" % ENERGYFILE
            
#SCRIPT
#get_tags()
#write_mood(2)
#write_mood_2()
#test_energy()
#test_loudness()
#print get_energy_feature('TRRABGI128E0780C8A')
print get_pitches('TRRABGI128E0780C8A')