'''
Created on 16 okt. 2012

@author: Erik Vandeputte
'''
from util import song_to_count
import cPickle as pickle

TRAIN_TRIPLETS_FILE = '../../msd_dense_subset/train_triplets_dense_subset.txt'
INTERACTION_MATRIX_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
SUBMISSION_FILE = '../../msd_dense_subset/recommendations_pop.txt'

#PARAMETERS
NUM_RECOMMENDATIONS = 50

#RESULTS
#map = 0.01295

stc = song_to_count(TRAIN_TRIPLETS_FILE)
print "loading interaction matrix"
with open(INTERACTION_MATRIX_FILE, 'r') as f:
    data = pickle.load(f)
    B = data['B']
    num_users = data['num_users']
    num_songs = data['num_songs']
    num_triplets = data['num_triplets']
    users_map = data['users_map']
    songs_map = data['songs_map'] #(songid, songnr)
    users_map_inv = data['users_map_inv']
    
popular_songs = sorted(stc, key=stc.get,reverse=True) 
#recommend for each user the most popular song that they haven't listened to yet
def generate_prediction_file():
    print "generating prediction file to %s" % SUBMISSION_FILE
    with open(SUBMISSION_FILE, 'w') as f:
        for i in range(num_users-1):
            songs_to_recommend = []
            for song in popular_songs:
                if len(songs_to_recommend) >= NUM_RECOMMENDATIONS:
                    break
                    # if the user hasnt listened to the song yet, add it.
                if not B[i, songs_map[song]]:
                    songs_to_recommend.append(song)
            f.write((users_map_inv[i])+' ' +' '.join(songs_to_recommend) + '\n')
    print "data saved to %s" % SUBMISSION_FILE
    
generate_prediction_file() 