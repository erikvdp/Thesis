'''
Created on 27 okt. 2012

@author: Erik Vandeputte
'''
import subevaluate
import numpy as np
import wmf_als

RESULTS_FILE = '../../msd_dense_subset/wmf_training.txt'


ev = subevaluate.Evaluator('../../msd_dense_subset/recommendations_wmf.txt')
ev.load_data() #load the validation data, has to be done only once
alphas = [100,250,500,750,1000]
epsilons = [10 ** -10,10 ** - 5, 10 ** -2,1,10]
regularizations_users = range(-3,4,1)
regularizations_songs = range(-3,4,1)
for alpha in alphas:
    for eps in epsilons:
        f = open(RESULTS_FILE, 'a')
        wmf_als.main(alpha,eps,1000,100)
        map = ev.evaluate_mAP()
        f.write("%s\t%s\t%s\t%s\t%s\n" % (str(alpha),str(eps),str(1000),str(100),str(map)))
        f.flush()
f.close()