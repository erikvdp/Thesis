'''
Created on 25 feb. 2013

@author: Erik Vandeputte
'''
import time
import numpy as np

TXTFILE='../mfcctxt/144.txt'
NPYFILE= '../mfccnpy/144.npy'

start_time = time.time()
np.genfromtxt(TXTFILE)
print "duration reading txt file = %f" % (time.time()-start_time)

start_time = time.time()
np.load(NPYFILE)
print "duration reading npy file = %f" %(time.time()-start_time)
