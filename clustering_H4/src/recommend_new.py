'''
Created on 8 apr. 2013
This script generates new recommendations based on the new V file
@author: Erik Vandeputte
'''
import cPickle as pickle
import numpy as np

FACTORS_FILE_1 ="../pklfiles/new_V_original_random.pkl"
FACTORS_FILE_2 ="../pklfiles/new_V_random_random.pkl"
FACTORS_FILE_3 ="../pklfiles/new_V_prediction_random.pkl"

SUBMISSION_FILE_1 = '../../msd_dense_subset/recommendations_wmf_original.txt'
SUBMISSION_FILE_2 = '../../msd_dense_subset/recommendations_wmf_random.txt'
SUBMISSION_FILE_3 = '../../msd_dense_subset/recommendations_wmf_prediction.txt'

INTERACTION_MATRIX_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
NUM_RECOMMENDATIONS = 50

def load_data(factors_file):
    global V,U,B,num_users,users_map_inv,songs_map_inv,test_idx
    with open(factors_file, 'r') as f:
        data = pickle.load(f)
        U = data['U']
        V = data['V_new']
        test_idx = data['test_idx'] #modified song factors
    with open(INTERACTION_MATRIX_FILE,'r') as f:
        data = pickle.load(f)
        B = data['B']
        num_users = data['num_users']
        users_map_inv = data['users_map_inv']
        songs_map_inv = data['songs_map_inv'] #(songnr, songid)


#generate predictions
def generate_prediction_file(submission_file):
    total = 0
    B_new = np.dot(U,V.T)
    print "generating prediction file to %s" % submission_file
    with open(submission_file, 'w') as f:
        for i in range(num_users-1):
            song_batch = np.argsort(B_new[i])[::-1] # sort and flip
            songs_to_recommend = []
            for song_idx in song_batch:
                if len(songs_to_recommend) >= NUM_RECOMMENDATIONS: 
                    break
                    # if the user hasnt listened to the song yet, add it.
                if not B[i, song_idx]:
                    if song_idx in test_idx:
                        total +=1
                    song = songs_map_inv[song_idx]
                    songs_to_recommend.append(song)
            f.write(users_map_inv[i]+' ' +' '.join(songs_to_recommend) + '\n')
    print "data saved to %s" % submission_file
    print "total predicted songs recommended %d" %total

load_data(FACTORS_FILE_1)
generate_prediction_file(SUBMISSION_FILE_1)
load_data(FACTORS_FILE_2)
generate_prediction_file(SUBMISSION_FILE_2)
load_data(FACTORS_FILE_3)
generate_prediction_file(SUBMISSION_FILE_3)