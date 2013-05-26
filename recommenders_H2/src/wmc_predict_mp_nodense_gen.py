import os
import cPickle as pickle
import numpy as np
import time
import multiprocessing as mp
import shutil

#EST RUNTIME = 760s

#RESULTS
#ALPHA = 0.5, Q = 6, N=-1  =>mAP:0.11759

def get_col_index_vector(m):
    """
    given an indptr vector of a CSR sparse matrix, compute a vector with length len(m.data)
    that gives the corresponding column index for each element.
    """
    num_rows = len(m.indptr) - 1
    num_elements = m.indptr[-1]
    vec = np.zeros((num_elements,), dtype='int32')
    
    for r in xrange(num_rows):
        start = m.indptr[r]
        end = m.indptr[r + 1]
        vec[start:end] = r
        
    return vec


def compute_song_scores(user_idx, P, PT, o_rescaled_x, o_rescaled_y, n, q):
    # get the songs that the user has listened to
    song_indices = P[user_idx].indices
    
    # Y: song the user has listened to
    # X: song that is a candidate for recommendation

    ### compute P(X,Y)
    # compute partial cooccurrence matrix C for the songs the user has listened to vs all other songs.
    cYX = PT[song_indices].dot(P) # Y x X
    
    denom_part1 = o_rescaled_y[cYX.indices]  # get a vector that matches the cYX data vector.
    denom_part2 = o_rescaled_x[song_indices][get_col_index_vector(cYX)] # same
    
    denom = (denom_part1 + denom_part2) ** (1.0 / n)
    
    cYX.data = (cYX.data / denom).astype('float32')
    cYX.data = cYX.data ** q
    
    return cYX.sum(0).A.ravel() # sum over Y





def predict(path, users, users_map, songs_map_inv, song_to_index, P, PT, o_rescaled_x, o_rescaled_y, n, q):
    with open(path, 'w') as f:
        for i, user in enumerate(users):
            user_idx = users_map[user]
            
            if (i+1) % 100 == 0:
                print "%s: %d of %d" % (path, i + 1, len(users))
            
            affinities = compute_song_scores(user_idx, P, PT, o_rescaled_x, o_rescaled_y, n, q)
            song_indices_user = np.argsort(affinities)[::-1] # sort and flip
            
            songs_to_recommend = []
            for song_idx in song_indices_user:
                if len(songs_to_recommend) >= NUM_RECOMMENDATIONS:
                    break
                # if the user hasnt listened to the song yet, add it.
                if not P[user_idx, song_idx]:
                    song = songs_map_inv[song_idx]
                    songs_to_recommend.append(song)

            f.write(user+' ' +' '.join(songs_to_recommend) + '\n')    
            del song_indices_user, affinities

def main(alpha,q,n):
    global NUM_RECOMMENDATIONS, ALPHA, Q, N
    # PARAMETERS
    ALPHA = alpha
    Q = q
    N = n
    
    INTERACTION_MATRIX_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
    submission_file = '../../msd_dense_subset/recommendations_wmc.txt'
    
    NUM_PROCESSES = 4
    NUM_RECOMMENDATIONS = 50
    print "Provided settings:"
    print "ALPHA=%s Q=%s N=%s" %(str(ALPHA),str(Q),str(N)) 

    print "load interaction matrix B"
    with open(INTERACTION_MATRIX_FILE, 'r') as f:
        data = pickle.load(f)
        B = data['B']
        num_users = data['num_users']
        num_songs = data['num_songs']
        num_triplets = data['num_triplets']
        users_map = data['users_map']
        users_map_inv = data['users_map_inv'] #(usernr, userid)
        songs_map_inv = data['songs_map_inv'] #(songnr, songid)
        songs_map = data['songs_map']
        

    print "construct preference matrix P"
    # P = B > 0 # preference matrix
    # ... doesn't work! We need to operate on the data again:
    P = B.copy()
    P.data = (P.data > 0).astype('int32')
    PT = P.T.tocsr()

    print "construct occurrence vector o" #??
    o = P.sum(0).A.ravel().astype('float32')
    
    print "construct rescaled occurrence vectors"
    o_rescaled_y = (o ** N) * ALPHA
    o_rescaled_x = (o ** N) * (1 - ALPHA)


    print "compute recommendations for the validation users"            
    # split the canonical users list into N chunks, start a process on each chunk
    num_users_per_thread = int(np.ceil(float(num_users) / NUM_PROCESSES))

    start_time = time.time()


    processes = []
    paths = []
    for i in range(NUM_PROCESSES):
        x  = users_map_inv.values()
        users_subset = x[i * num_users_per_thread:(i + 1) * num_users_per_thread] #list with userids
        path = "%s.part%d" % (submission_file, i)
        paths.append(path)
        p = mp.Process(target=predict, args=(path, users_subset, users_map, songs_map_inv, songs_map, P, PT, o_rescaled_x, o_rescaled_y, N, Q))
        processes.append(p)
        p.start()


    # wait for all processes to complete
    for p in processes:
        p.join()

    print "  generating predictions took %.2f seconds" % (time.time() - start_time)

    # merge all the partial output files into one
    print "merging output file parts"
    destination = open(submission_file, 'w')
    for path in paths:
        shutil.copyfileobj(open(path, 'r'), destination)
    destination.close()

    print "removing output file parts"
    for path in paths:
        os.remove(path)

    print "  predictions saved to %s" % submission_file
        







