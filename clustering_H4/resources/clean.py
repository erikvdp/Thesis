import urllib
import time
import random
import os
import sys
import platform
import os
import shutil
import time

path = '/mnt/storage/data/msd/'
# path = ''

for i in range(10):
    for j in range(10):
        directory = path + 'songs/' + str(i) + '/' + str(j)
        if not os.path.exists(directory):
            os.makedirs(directory)

dirList=os.listdir(path + 'songs')
for fname in dirList:
    if fname.endswith('.mp3'):
        i, j = fname[0], fname[1]
        try:
        	shutil.move(path + 'songs/' + fname, path + 'songs/' + str(i) + '/' + str(j) + '/' + fname)
        except:
        	continue
        # time.sleep(0.1)
