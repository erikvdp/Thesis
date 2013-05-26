'''
Created on 16 feb. 2013

@author: Erik Vandeputte
'''
import os

IDFILE = "problemids.txt"
i = 1
with open(IDFILE, 'r') as f1:
    for line in iter(f1):
        digitalid = line
        print digitalid
        command = "scp -P8834 emvdputt@localhost:/mnt/storage/data/msd/songs/"+digitalid[0]+"/"+digitalid[1]+"/"+digitalid.strip()+".clip.mp3 ./songs/"
        print command
        print i
        os.system(command)
        i = i+1
f1.close()

    
