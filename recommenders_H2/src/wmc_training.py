'''
Created on 30 okt. 2012

@author: Erik Vandeputte
'''

import subevaluate
import numpy as np
import wmc_predict_mp_nodense_gen

RESULTS_FILE = '../../msd_dense_subset/wmc_training.txt'


ev = subevaluate.Evaluator('../../msd_dense_subset/recommendations_wmc.txt')
ev.load_data() #load the validation data, has to be done only once
alphas = [0.2, 0.3, 0.35,0.4]
qs = range(4,5)
ns = [1]
for q in qs:
    for alpha in alphas:
            f = open(RESULTS_FILE, 'a')
            wmc_predict_mp_nodense_gen.main(alpha,q,1)
            map = ev.evaluate_mAP()
            f.write("%s\t%s\t%s\t%s\n" % (str(alpha),str(q),str(1),str(map)))
            f.flush()
f.close()