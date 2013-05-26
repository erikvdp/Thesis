"""
Evaluate on the subeval set, where part 2 is 'unseen'.
Compute truncated mAP.
Compute own 
Based on the explanation in section 4.2 of the MSD challenge paper
"""

import numpy as np
import cPickle as pickle
class Evaluator:

    INTERACTION_MATRIX_FILE = '../../msd_dense_subset/interaction_matrix.pkl'
    VALIDATION_TRIPLETS_FILE = '../../msd_dense_subset/validation_triplets_dense_subset.txt'
    users_songs = {}

    def get_submission_file(self):
        return self.submission_file

    def set_submission_file(self, value):
        self.submission_file = value

    
    def __init__(self,submission_file):
        self.submission_file = submission_file
    
    def load_data(self):
        print "load validation data"
        with open(self.VALIDATION_TRIPLETS_FILE, 'r') as f:
            for line in f:
                user, song,_ = line.strip().split("\t")
                if user not in self.users_songs:
                    self.users_songs[user] = set()
                self.users_songs[user].add(song)
                  
    def evaluate_mAP(self):
        with open(self.submission_file, 'r') as f:
                avg_precisions = []
                np.seterr(all='raise')
                for line in f:
                    data = line.strip().split(" ")
                    user = data[0]
                    songs = data[1:]
                    tau = len(songs) #tau = number of recommendations made
                    if user not in self.users_songs.keys() or tau == 0: #the user is not in the validation file or we have no recommendations for the user
                        continue
                    n_u = min(tau, len(self.users_songs[user])) # the smaller of tau and the number of positively associated songs for the user
                    hits = np.array([song in self.users_songs[user] for song in songs]).astype('float64') # this corresponds to M[u,i] from the paper&
                    precisions = np.cumsum(hits) / np.arange(1, tau + 1) # precision at k for every k, these are the Pk, formula (1) in the paper
                    avg_precision = np.dot(precisions, hits) / n_u # formula (2) in the paper
                    avg_precisions.append(avg_precision)
        mAP = np.mean(avg_precisions) # formula (3) in the paper
        
        print "  The mAP is %.5f for %s" % (mAP,self.submission_file)
        return mAP
    
    def evaluate_novelty(self):
        with open(self.INTERACTION_MATRIX_FILE,'r') as f:
            data = pickle.load(f)
            self.song_plays = data['song_plays']
            del data
        with open(self.submission_file, 'r') as f:
                avg_novelties = []
                for line in f:
                    data = line.strip().split(" ")
                    user = data[0]
                    songs = data[1:]
                    tau = len(songs)
                    if user not in self.users_songs.keys() or tau == 0: #the user is not in the validation file or we have no recommendations for the user
                        continue
                    n_u = tau # the number of recommendations made for the user
                    total_plays = np.array([self.song_plays[song] for song in songs]).astype('float64') # this corresponds to M[u,i] from the paper&
                    novelty = (n_u)/np.log(sum(total_plays))
                    avg_novelties.append(novelty)
        nov = np.mean(avg_novelties) # formula (3) in the paper
        
        print "  The novelty is %.5f" % nov
        return nov
    
    def evaluate_recall(self): #calculate recall in each point 1-50
        print "evaluating recalls for %s" %self.submission_file
        with open(self.submission_file, 'r') as f:
                avg_recalls = list()
                for line in f:
                    data = line.strip().split(" ")
                    user = data[0]
                    songs = data[1:]
                    tau = len(songs) #tau = number of recommendations made
                    if user not in self.users_songs.keys() or tau == 0: #the user is not in the validation file or we have no recommendations for the user
                        continue
                    n_u = min(tau, len(self.users_songs[user])) # the smaller of tau and the number of positively associated songs for the user
                    hits = np.array([song in self.users_songs[user] for song in songs]).astype('float64') # this corresponds to M[u,i] from the paper&
                    recalls = np.cumsum(hits) / (np.ones(tau)*n_u) #recall at each position k 
                    avg_recalls.append(recalls)
        avg_recalls = np.array(avg_recalls)
        avg_recalls = np.mean(avg_recalls,axis=0)
        print "the average recall at point 50 is %.5f" %np.mean(avg_recalls)
        return avg_recalls
        
#SCRIPT
ev = Evaluator("../../msd_dense_subset/recommendations_sameartist.txt")
ev.load_data()
ev.evaluate_mAP()
ev.evaluate_novelty()
ev.evaluate_recall()