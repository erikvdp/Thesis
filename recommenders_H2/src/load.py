'''
Created on 9 okt. 2012
Parsing training data and store it in %TARGET_TRAINING_FILE
@author: Erik Vandeputte
'''

from scipy import sparse
import numpy as np
import cPickle as pickle


TARGET_TRAINING_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
TRAIN_TRIPLETS_FILE = '../../msd_dense_subset/train_triplets_dense_subset.txt'
VALIDATION_TRIPLETS_FILE = '../../msd_dense_subset/validation_triplets_dense_subset.txt'

# run through the triplets files a first time to gather songs and users and map them to integers
users_set = set() # set of all seen user IDs, for fast lookup
songs_set = set() # set of all seen song IDs, for fast lookup
users_map = {} # map the long user IDs to integers
users_map_inv = {} # map the integer IDs to username
songs_map = {} # map the long song IDs to integers
songs_map_inv = {} # map the integer IDs to full song IDs
users_songs = dict() #list of listened songs for each user
song_plays = dict() #number of plays for each song

def load_triplets(filename):
    with open(filename, 'r') as f:
        for line in f:
            user, song, _ = line.strip().split('\t')
            if user not in users_set:
                new_user_id = len(users_set) # assign new integer ID to this user
                users_map[user] = new_user_id
                users_map_inv[new_user_id] = user
                users_set.add(user)
                users_songs.setdefault(user,set()).add(song)
            else:
                users_songs[user].add(song)
            if song not in songs_set:
                new_song_id = len(songs_set) # assign new integer ID to this song
                songs_map[song] = new_song_id
                songs_map_inv[new_song_id] = song
                songs_set.add(song)
                song_plays[song] = 0
def populate_matrix(filename):
    with open(filename, 'r') as f:
        for line in f:
            user, song, count = line.strip().split('\t')
            user_int = users_map[user];
            song_int = songs_map[song];
            A[user_int, song_int] = int(count)
            song_plays[song] += int(count)

print "load the triplets"
load_triplets(TRAIN_TRIPLETS_FILE)

num_users = len(users_set)
num_songs = len(songs_set)

A = sparse.lil_matrix((num_users, num_songs), dtype='int32')

print "populate interaction matrix"
populate_matrix(TRAIN_TRIPLETS_FILE)

B = A.tocsr().astype("float32") #conversion to a compressed sparse row matrix
nz_u, nz_i = B.nonzero() # get indices for nonzero elements
num_triplets = len(nz_u)

#get the indices for the least played songs (count = number of users that listened)
idx_least_to_most = np.argsort(np.bincount(np.nonzero(A)[1]))
#get the songid's fot the least played songs (count = numbers of users that listened)
songid_least_to_most = list()
for idx in idx_least_to_most:
    songid_least_to_most.append(songs_map_inv[idx])
songid_least_to_most = np.array(songid_least_to_most)

print num_triplets
print "pickle data"

data = {
    'B': B,
    'num_users': num_users,
    'num_songs': num_songs,
    'num_triplets': num_triplets,
    'users_map': users_map,
    'songs_map': songs_map,
    'songs_map_inv': songs_map_inv,
    'users_map_inv': users_map_inv,
    'users_songs': users_songs,
    'song_plays':song_plays,
    'idx_least_to_most':idx_least_to_most,
    'songid_least_to_most':songid_least_to_most
}

with open(TARGET_TRAINING_FILE, 'w') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
print "data saved to %s" % TARGET_TRAINING_FILE

