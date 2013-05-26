'''
Created on 27 apr. 2013

@author: Erik Vandeputte
'''
import cPickle as pickle
import time
from itertools import izip
import numpy as np

INTERACTION_MATRIX_FILE = './pklfiles/training_set.pkl'
NEW_INTERACTION_MATRIX_FILE = './pklfiles/factormatrices.pkl'

#PARAMETERS
ALPHA = 20 #rate of incrase for the surplus confidence matrix
EPSILON = 1 # 10 ** (-3)
NUM_ITERATIONS = 15
REGULARISATION_USERS = 1000
REGULARISATION_SONGS = 100
INIT_STD = 0.1
NUM_FACTORS = 50

#load interaction matrix
def load_interaction_matrix():
    global B, num_users, num_songs, num_triplest, songs_map, users_map, songs_map_inv,users_map_inv,num_training_songs
    print "loading interaction matrix"
    with open(INTERACTION_MATRIX_FILE, 'r') as f:
        data = pickle.load(f)
        B = data['B']
        num_users = data['num_users']
        num_songs = data['num_songs']
        num_training_songs = data['num_training_songs']
        songs_map = data['songs_map']
        users_map = data['users_map']
        users_map_inv = data['users_map_inv']
        songs_map_inv = data['songs_map_inv'] #(songnr, songid)

def prepare_factorisation():
    global U,V,Sl,Dl,Il,STl, DTl, ITl
    print "precompute matrices"
    start_time = time.time() 
    # Constructing the surplus confidence matrix is tricky: log(1 + B) can't be done in one go, because
    # 1 + B is of course impossible (not sparse). We need to operate directly on the nonzero elements of the sparse
    # matrix B.
    # S = ALPHA * np.log(1 + B / EPSILON) # surplus confidence matrix # this is not possible
    print "  surplus confidence matrix S and its transpose ST"
    S = B.copy()
    S.data = ALPHA * np.log(1 + S.data / EPSILON)
    ST = S.T.tocsr()
    print "  D = (S+1)*P where P is the preference matrix, and its transpose DT"
    # Constructing the preference matrix seems to be tricky as well, because this:
    # P = B > 0 # preference matrix
    # ... doesn't work! We need to operate on the data again:
    P = B.copy()
    P.data = (P.data > 0)

    # Construct D on beforehand to avoid duplicate computation. D = P*S + P, which is equal to P*C
    D = P.multiply(S) + P
    DT = D.T.tocsr()
    del P # won't need this anymore

    Sl = np.empty((D.shape[0]), dtype='object')
    Dl = np.empty((D.shape[0]), dtype='object')
    Il = np.empty((D.shape[0]), dtype='object')

    STl = np.empty((DT.shape[0]), dtype='object')
    DTl = np.empty((DT.shape[0]), dtype='object')
    ITl = np.empty((DT.shape[0]), dtype='object')

    print "  convert S and D for faster row iteration"
    for i, s_u, d_u in izip(xrange(D.shape[0]), S, D):
        if i % 10000 == 0:
            print "    at row %d" % i
        Dl[i] = d_u.data
        Sl[i] = s_u.data
        Il[i] = d_u.indices
    
    print "  convert ST and DT for faster row iteration"
    for i, s_u, d_u in izip(xrange(DT.shape[0]), ST, DT):
        if i % 10000 == 0:
            print "    at row %d" % i
        DTl[i] = d_u.data
        STl[i] = s_u.data
        ITl[i] = d_u.indices
    
    del DT, D, ST, S # don't need these anymore

    print "  took %.2f seconds" % (time.time() - start_time)


    print "start ALS training"

    U = np.random.randn(num_users, NUM_FACTORS).astype('float32') * INIT_STD # technically obsolete, but I like symmetry.
    V = np.random.randn(num_training_songs, NUM_FACTORS).astype('float32') * INIT_STD

def recompute_factors(Y, Sl, Dl, Il, lambda_reg):
    """
    recompute matrix X from Y.
    X = recompute_factors(Y, Sl, Dl, Il, lambda_reg_x)
    This can also be used for the reverse operation as follows:
    Y = recompute_factors(X, STl, DTl, ITl, lambda_reg_y)
    
    Sl: data for S matrix
    Dl: data for D matrix
    Il: indices for both matrices
    
    The comments are in terms of X being the users and Y being the items.
    """
    m = Dl.shape[0] # m = number of users
    f = Y.shape[1] # f = number of factors
    YTY = np.dot(Y.T, Y) # precompute this
    YTYpI = YTY + lambda_reg * np.eye(f)
    X_new = np.zeros((m, f), dtype='float32')
    for k, s_u, d_u, i_u in izip(xrange(m), Sl, Dl, Il):
        # if k % 1000 == 0:
        #    print "%d" % k
        Y_u = Y[i_u] # exploit sparsity
        A = d_u.dot(Y_u)
        YTSY = np.dot(Y_u.T, (Y_u * s_u.reshape(-1, 1)))
        B = YTSY + YTYpI
        # Binv = np.linalg.inv(B)
        # X_new[k] = np.dot(A, Binv) 
        X_new[k] = np.linalg.solve(B.T, A.T).T # doesn't seem to make much of a difference in terms of speed, but w/e
    return X_new

def factorisation():
    global B_new,V,U,Sl,Dl,Il,STl, DTl, ITl
    start_time = time.time()
    for i in range(NUM_ITERATIONS):
        print "ITERATION %d" % i
        
        print "  recompute song factors V"
        V_new = recompute_factors(U, STl, DTl, ITl, REGULARISATION_SONGS)
        rms_dV = np.sqrt(np.mean((V - V_new) ** 2))
        print "  RMS difference %.4f" % rms_dV
        print "  Time since start: %.2f seconds" % (time.time() - start_time)
        V = V_new

        print "  recompute user factors U"
        U_new = recompute_factors(V, Sl, Dl, Il, REGULARISATION_USERS)
        rms_dU = np.sqrt(np.mean((U - U_new) ** 2))
        print "  RMS difference %.4f" % rms_dU
        print "  Time since start: %.2f seconds" % (time.time() - start_time)
        U = U_new
    

    del Sl, Dl, Il, STl, DTl, ITl # don't need these anymore

    print "We have now user and song factors : )"
    print "store the resulting user_item matrix"
    B_new = np.dot(U,V.T)
    
def save_data():
    print "pickle data"

    data = {
            'songs_map': songs_map,#(songid, songnr)
            'songs_map_inv': songs_map_inv, #(songnr, songid)
            'V': V,
            'U':U,
            'B_new':B_new,
            'B_old':B
            }

    with open(NEW_INTERACTION_MATRIX_FILE, 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print "data saved to %s" % NEW_INTERACTION_MATRIX_FILE
    print "pickle data"

load_interaction_matrix()
prepare_factorisation()
factorisation()
save_data()
