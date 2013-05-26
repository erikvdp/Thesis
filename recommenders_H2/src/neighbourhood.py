'''
!!!! 
DEPRECATED

see wmc_predict_mp_nodense_gen.py for faster implementation

!!!
Created on 20 okt. 2012

@author: Erik Vandeputte
'''
import cPickle as pickle
import numpy as np
import math
import time
#TODO: take account implicit feedback?


INTERACTION_MATRIX_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
TARGET_FILE = '../../msd_dense_subset/similarity_matrix.pkl'
SUBMISSION_FILE = '../../msd_dense_subset/recommendations_neighbourhood.txt'

#PARAMETERS
NUM_RECOMMENDATIONS = 50
def load_data():
    #load interaction matrix
    print "loading interaction matrix"
    with open(INTERACTION_MATRIX_FILE, 'r') as f:
        global B,num_users,num_songs,num_triplets,users_map,songs_map_inv
        data = pickle.load(f)
        B = data['B'] #(user,song)
        num_users = data['num_users']
        num_songs = data['num_songs']
        num_triplets = data['num_triplets']
        users_map = data['users_map']
        songs_map_inv = data['songs_map_inv'] #(songnr, songid)

def pre_compute_similarity():
    global users,avg
    start_time = time.time()
    #compute similarity matrix
    print "compute average rating for each song (item-based)"
    #construct average score array
    avg = np.empty(num_songs, dtype='object')
    for i in range(num_songs):
        avg[i] = sum(B[:,i].data)/len(B[:,i].data)
    #construct dict of users who listened to a certain song
    print "compute list of users for each song"
    users = np.empty(num_songs, dtype='object')
    for i in range(num_songs):
        users[i] = set(np.nonzero(B[:,i])[0])
    print "  took %.2f seconds" % (time.time() - start_time)
  
def compute_similarity_old():
    print "construct correlation matrix"
    start_time = time.time() 
    co = np.empty((num_songs,num_songs), dtype='object')    
    for i in range(num_songs): #pearson correlation is symmetric
        print i
        for j in range(i+1,num_songs-1):
            print j
            rating_left = B[:,i].todense()
            rating_right= B[:,j].todense()
            rating_left = np.squeeze(np.asarray(rating_left))
            rating_right= np.squeeze(np.asarray(rating_right))
            common = np.logical_and(rating_left > 0, rating_right > 0)
            common = np.squeeze(np.asarray(common))
            if sum(common) ==0:
                co[i,j] = 0
                co[j,i] = 0
            else:
                numerator = sum((rating_left[common]-avg[i]) * (rating_right[common]-avg[j]))
                denominator = math.sqrt(sum((rating_left[common]-avg[i])**2))*math.sqrt(sum((rating_right[common]-avg[j])**2))
            if denominator < 0.0001:
                co[i,j] = 0
                co[j,i] = 0
            else:         
                r = numerator / denominator
                co[i,j] = r
                co[j,i] = r
    print "  took %.2f seconds" % (time.time() - start_time)
#construct pearson correlation between each item
def compute_similarity():
    print "construct correlation matrix"
    co = np.empty((num_songs,num_songs), dtype='object')    
    start_time = time.time() 
    for i in range(num_songs):
        print i
        for j in range(i+1,num_songs):
            #get the set of users who rated items i and j use fast intersection function
            cousers = list(users[i] & users[j])
            if len(cousers) == 0:
                continue
            nominator = sum(np.multiply((B[cousers,i].todense() - avg[i]),(B[cousers,j].todense() - avg[j])))
            denominator = np.multiply(math.sqrt(sum(np.power(B[cousers,i].todense()-avg[i],2))),math.sqrt(sum(np.power(B[cousers,j].todense()-avg[j],2))))
            if denominator < 0.0001:
                co[i][j] = 0
                co[j][i] = 0
            else:  
                co[i][j] =  nominator / denominator
                co[i][j] =  nominator / denominator
    print "  took %.2f seconds" % (time.time() - start_time)

  
def save_similarity_data():
    #pickle some stuff
    print "pickle data"
    data = {
            'avg': avg,
            'users':users,
            'co':co
            }

    with open(TARGET_FILE, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    print "data saved to %s" % TARGET_FILE

def load_similarity_data():
    global co,users,avg
    #load similarity data
    print "loading cosimilarity matrix"
    with open(TARGET_FILE, 'r') as f:
        data = pickle.load(f)
        avg = data['avg']
        users = data['users']

def generate_predicition():
    #generate prediction matrix TODO
    global B_new
    B_new = np.empty((num_users,num_songs),dtype='object')
    for i in range(num_users):
        for j in range(num_songs):
            similar_songs_indices = ()
            nominator = sum(*B[i,similar_songs_indices])
            denominator = sum()
            B_new[i,j] = nominator / denominator
            
def generate_prediction_file():
    print "generating prediction file to %s" % SUBMISSION_FILE
    with open(SUBMISSION_FILE, 'w') as f:
        for user in range(num_users-1):
            song_batch = np.argsort(B_new[user])[::-1] # sort and flip
            songs_to_recommend = []
            for song_idx in song_batch:
                if len(songs_to_recommend) >= NUM_RECOMMENDATIONS: 
                    break
                    # if the user hasnt listened to the song yet, add it.
                if not B[user, song_idx]:
                    song = songs_map_inv[song_idx]
                    songs_to_recommend.append(song)
            f.write(' '.join(songs_to_recommend) + '\n')
    print "data saved to %s" % SUBMISSION_FILE

#SCRIPT
 
load_data()
'''
pre_compute_similarity()
save_similarity_data()
'''
load_similarity_data()
compute_similarity()