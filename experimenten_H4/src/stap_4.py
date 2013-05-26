'''
Created on 27 apr. 2013

@author: Erik Vandeputte
'''
import cPickle as pickle
import scipy.sparse
import numpy as np
import time
from itertools import izip


TEST_FILE = './pklfiles/test_set.pkl'
FACTORS_FILE = './pklfiles/factormatrices.pkl'
AUDIO_FACTORS_FILE = './pklfiles/V_test_audio.pkl'

PREDICTIONS_FILE = './pklfiles/predictions_'


#PARAMETERS WMF
ALPHA = 20 #rate of incrase for the surplus confidence matrix
EPSILON = 0.5 # 10 ** (-3)
NUM_ITERATIONS = 20
REGULARISATION_USERS = 0.01
REGULARISATION_SONGS = 0.01
INIT_STD = 0.1
NUM_FACTORS = 50


def load_data():
    global X_test,U,V_test_audio,songs_map_inv,users_map_inv,B
    with open(TEST_FILE,'r') as f:
        data = pickle.load(f)
        B = data['B']
        X_test = np.asarray(data['B'].todense())
        songs_map_inv = data['songs_map_inv']
        users_map_inv = data['users_map_inv']
    with open(FACTORS_FILE,'r') as f:
        data = pickle.load(f)
        U = data['U']
    with open(AUDIO_FACTORS_FILE) as f:
        V_test_audio = pickle.load(f)
        
def remove_listening_data(alpha): #remove the dumb way, alpha % listeners in test set
    global X_test_A, X_test_B,remove_idx,non_remove_idx
    new_alpha = X_test.shape[0] * alpha
    print new_alpha
    random_idx = np.random.permutation(X_test.shape[0])
    remove_idx = random_idx[0:new_alpha]
    non_remove_idx = random_idx[new_alpha]
    X_test_A = np.copy(X_test)
    X_test_B = np.copy(X_test)
    X_test_B[remove_idx,:] = 0 #het cold start probleem (weinig informatie per nummer)
    X_test_A[non_remove_idx:,:] = 0
    
    X_test_B = scipy.sparse.csr_matrix(X_test_B) #convert X_test_B to sparse matrix  

def remove_listening_data_smart(alpha): #remove the smart way, alpha % play counts in test set for each song
    global X_test_A, X_test_B
    X_test_A = np.copy(X_test)
    X_test_B = np.copy(X_test)
    for i in range(X_test.shape[1]): 
        non_zero_idx = X_test[:,i].nonzero()[0]
        non_zero_idx = np.random.permutation(non_zero_idx)
        new_alpha = np.floor(len(non_zero_idx)*alpha)
        train_zero_idx = non_zero_idx[:new_alpha]
        test_zero_idx = non_zero_idx[new_alpha:]
        X_test_A[test_zero_idx,i] = 0
        X_test_B[train_zero_idx,i] = 0
    X_test_B = scipy.sparse.csr_matrix(X_test_B) #convert X_test_B to sparse matrix
    print "number of entries in training data: %d" %len(np.nonzero(X_test_B)[0])
    
def remove_listening_data_smart_fixed(alpha_fixed):  #remove the smart way, alpha play counts in training set for each song
    global X_test_A, X_test_B
    X_test_A = np.copy(X_test)
    X_test_B = np.copy(X_test)
    for i in range(X_test.shape[1]): 
        non_zero_idx = X_test[:,i].nonzero()[0]
        non_zero_idx = np.random.permutation(non_zero_idx)
        if(len(non_zero_idx) > alpha_fixed):
            train_zero_idx = non_zero_idx[alpha_fixed:]
            test_zero_idx = non_zero_idx[:alpha_fixed]
        else:
            train_zero_idx = []
            test_zero_idx = non_zero_idx
        X_test_A[test_zero_idx,i] = 0
        X_test_B[train_zero_idx,i] = 0
    X_test_B = scipy.sparse.csr_matrix(X_test_B) #convert X_test_B to sparse matrix
    print "number of entries in training data: %d" %len(np.nonzero(X_test_B)[0])
    
def matrix_factorization():
    #(Optimaliseer V_test_mf gegeven U zo dat:  U * V_test_mf ~= X_test_B)
    global V_test_mf
    prepare_factorisation()
    print "  recompute song factors V"
    V_test_mf = recompute_factors(U, STl, DTl, ITl, REGULARISATION_SONGS)
    rms_dV = np.sqrt(np.mean((V - V_test_mf) ** 2))
    print "  RMS difference %.4f" % rms_dV

def prepare_factorisation():
    global U,V,Sl,Dl,Il,STl, DTl, ITl
    print "precompute matrices"
    start_time = time.time() 
    # Constructing the surplus confidence matrix is tricky: log(1 + B) can't be done in one go, because
    # 1 + B is of course impossible (not sparse). We need to operate directly on the nonzero elements of the sparse
    # matrix B.
    # S = ALPHA * np.log(1 + B / EPSILON) # surplus confidence matrix # this is not possible
    print "  surplus confidence matrix S and its transpose ST"
    S = X_test_B.copy()
    S.data = ALPHA * np.log(1 + S.data / EPSILON)
    ST = S.T.tocsr()
    print "  D = (S+1)*P where P is the preference matrix, and its transpose DT"
    # Constructing the preference matrix seems to be tricky as well, because this:
    # P = B > 0 # preference matrix
    # ... doesn't work! We need to operate on the data again:
    P = X_test_B.copy()
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

    V = np.random.randn(1943, NUM_FACTORS).astype('float32') * INIT_STD

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


def recommend():
    global pred_audio, pred_mf

    pred_audio = np.dot(U,V_test_audio.T)
    
    pred_mf = np.dot(U,V_test_mf.T)
    
    pred_mf = (pred_mf + 1000.) * (np.asarray(X_test_B.todense()) == 0)
    pred_audio = (pred_audio + 1000.) * (np.asarray(X_test_B.todense()) == 0)
   
def save_data(alpha):
    
    data = {
    'X_test_A' : X_test_A,
    'pred_audio': pred_audio,
    'pred_mf': pred_mf,
    }

    with open(PREDICTIONS_FILE+str(alpha)+'.pkl', 'w') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
    print "data saved to %s" % PREDICTIONS_FILE+str(alpha)+'.pkl'

def main():
    for alpha in [100,50,20,10,5,3,2,1]:
        load_data()
        remove_listening_data_smart_fixed(alpha)
        matrix_factorization()
        recommend()
        save_data(alpha)
    print " done"