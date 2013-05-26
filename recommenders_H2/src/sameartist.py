'''
Created on 18 okt. 2012

!!!

ALGORITHM : recommend for each user the most played songs for an artist that they've played
note: run map_artists_for_users() after creating a training and validation set

!!!
@author: Erik Vandeputte
'''
from util import song_to_count
import cPickle as pickle
from util import song_to_artist
import hdf5_getters as GETTERS


TRAIN_TRIPLETS_FILE = '../../msd_dense_subset/train_triplets_dense_subset.txt'
INTERACTION_MATRIX_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
SUBMISSION_FILE = '../../msd_dense_subset/recommendations_sameartist.txt'
USERS_ARTIST_FILE = '../../msd_dense_subset/users_artists.pkl'

#PARAMETERS
NUM_RECOMMENDATIONS = 50


#RESULTS
#max map = 0.02905

#returns a dict with (user, artists)
def map_artists_for_users():
    users_artists = dict()
    songs_tracks = pickle.load(open ("../msd_dense_subset/dense/songs_tracks.pkl",'r'));
    for user in users_songs:
        print user
        users_artists[user] = set()
        for song in users_songs[user]:
            track = str(songs_tracks[song])
            # build path
            path = "../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
            h5 = GETTERS.open_h5_file_read(path)
            users_artists[user].add(GETTERS.get_artist_name(h5))
            h5.close()
    #store in pickle file for the moment
    with open(USERS_ARTIST_FILE, 'w') as f:
        pickle.dump(users_artists, f, pickle.HIGHEST_PROTOCOL)
        print "data saved to %s" % USERS_ARTIST_FILE
        
#map the most popular songs to their artists
def songs_artists (popularsongs):
    print "mapping 5000 most popular songs to their artists"
    popularsongs_artists = dict()
    for i in range(10000):
        popularsongs_artists[popularsongs[i]] = song_to_artist(popularsongs[i])
    return popularsongs_artists

#recommend for each user the most played song for an artist that they've played
def generate_prediction_file():
    users_artists = pickle.load(open (USERS_ARTIST_FILE,'r'))
    users_map_inv = dict((v,k) for k, v in users_map.iteritems())
    print "generating prediction file to %s" % SUBMISSION_FILE
    with open(SUBMISSION_FILE, 'w') as f:
        for i in range(num_users-1):
            songs_to_recommend = []
            for j in range(10000):
                song = popular_songs[j]
                artist = popularsongs_artists[song]
                if len(songs_to_recommend) >= NUM_RECOMMENDATIONS:
                    break
                    # if the user hasnt listened to the song yet but knows the artist => add it.
                if not B[i, songs_map[song]] and artist in users_artists[users_map_inv[i]]:
                    songs_to_recommend.append(song)
            f.write((users_map_inv[i])+' ' +' '.join(songs_to_recommend) + '\n')
    print "data saved to %s" % SUBMISSION_FILE


#SCRIPT
    
print "loading interaction matrix"
with open(INTERACTION_MATRIX_FILE, 'r') as f:
    data = pickle.load(f)
    B = data['B']
    num_users = data['num_users']
    num_songs = data['num_songs']
    num_triplets = data['num_triplets']
    users_map = data['users_map']
    users_map_inv = data['users_map_inv']    
    songs_map = data['songs_map'] #(songid, songnr)
    users_songs = data['users_songs']


stc = song_to_count(TRAIN_TRIPLETS_FILE)
popular_songs = sorted(stc, key=stc.get,reverse=True)
popularsongs_artists = songs_artists(popular_songs)
generate_prediction_file()
'''
map_artists_for_users()'''