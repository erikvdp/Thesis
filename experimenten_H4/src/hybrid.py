'''
Created on 29 apr. 2013

@author: Erik Vandeputte
'''
import numpy as np
import cPickle as pickle
from sklearn.metrics import roc_curve, auc

PREDICTIONS_FILE ="./pklfiles/predictions_"

RESULTS_FILE_mAP = "results_beta_mAP.txt"
RESULTS_FILE_AUC = "results_beta_AUC.txt"

def load_data(alpha):
    global pred_audio, pred_mf,pred_hybrid,truth
    with open(PREDICTIONS_FILE+str(alpha)+'.pkl','r') as f:
        data = pickle.load(f)
        pred_audio = data['pred_audio']
        pred_mf = data['pred_mf']
        pred_hybrid = pred_audio + pred_mf
        truth = data['X_test_A']
    truth[truth > 0] =1 #set all to one or zero
        

def calc_mAP(preds,cut_off=500):
    sorting_indices = np.argsort(-preds, 1)
    n = truth.shape[0]
    sorted_preds = (preds[np.arange(n), sorting_indices.T].T)[:, :cut_off]
    hits = (truth[np.arange(n), sorting_indices.T].T)[:, :cut_off]

    n_u = np.minimum(cut_off, np.sum(truth, 1))
    valid_idx = n_u.nonzero()[0]
    precisions = np.cumsum(hits, 1) / np.arange(1, cut_off + 1).reshape((1, cut_off))
    avg_precision = np.sum(precisions[valid_idx] * hits[valid_idx], 1) / n_u[valid_idx]
    print 'mAP: %f' %np.mean(avg_precision)
    return np.mean(avg_precision)
         
def calc_AUC(predictions):
    roc_aucs = list()
    for i in range(1000,5000): 
        #if i % 1000 == 0 and i != 0:
        #    print '%d users done' %i
        prediction = predictions[i,:]
        if sum(truth[i]) > 0: #numbers for this user in test set
            fpr, tpr, thresholds = roc_curve(truth[i],prediction)
            roc_aucs.append(auc(fpr, tpr))
    print 'AUC: %f' %np.mean(roc_aucs)
    return np.mean(roc_aucs)
    
def evaluate_test():
    with open(RESULTS_FILE_mAP,'a') as f1, open(RESULTS_FILE_AUC,'a') as f2:
        for alpha in [0.9,0.95,0.99,0.995,0.999]:
            load_data(alpha)
            f1.write('ALPHA=%f \n' %(alpha))
            f2.write('ALPHA=%f \n' %(alpha))
            for beta in [.25]:
                pred_hybrid = (1- beta)* pred_audio + (beta) * pred_mf
                f1.write('beta %f \t mAP \t %f\n' %(beta,calc_mAP(pred_hybrid)))
                f2.write('beta %f \t AUC \t %f\n' %(beta,calc_AUC(pred_hybrid)))
    f1.close()
    f2.close()
evaluate_test()