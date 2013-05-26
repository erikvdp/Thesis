'''
Created on 14 nov. 2012
This module provides code to deteremine appropiate features
this uses the mutual information metric
@author: Erik Vandeputte
'''

import util as ut
import hdf5_getters as GETTERS
import numpy as np
from scipy import stats
from sklearn import metrics
from pylab import *

SOURCE_DATA_FILE = "../../msd_dense_subset/mood2.txt"

NUM_FEATURES = 139

tracks = list()
labels = list()
with open(SOURCE_DATA_FILE, 'r') as f:
    for line in f:
        track, label = line.strip().split()
        if int(label) != -2:
            tracks.append(track)
            labels.append(label)
modes = np.zeros(len(tracks))
keys = np.zeros(len(tracks))
loudnesses = np.zeros((len(tracks)))
loudnesses_interval = np.zeros((len(tracks)))
loudnesses_var = np.zeros((len(tracks)))
tempos = np.zeros((len(tracks)))
time_signatures = np.zeros(len(tracks))
energies = np.zeros(len(tracks))

timbre_means = np.zeros((len(tracks),12))
timbre_vars = np.zeros((len(tracks),12))
timbre_median = np.zeros((len(tracks),12))
timbre_min = np.zeros((len(tracks),12))
timbre_max = np.zeros((len(tracks),12))
pitches_means = np.zeros((len(tracks),12))
pitches_vars = np.zeros((len(tracks),12))
pitches_median = np.zeros((len(tracks),12))
pitches_min = np.zeros((len(tracks),12))
pitches_max = np.zeros((len(tracks),12))

for idx,track in enumerate(tracks):
    h5 = GETTERS.open_h5_file_read("../../msd_dense_subset/mood/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5")    #fetch h5 file to allow faster preprocessing
    keys[idx], modes[idx]= ut.get_key_feature(track,h5)
    loudnesses[idx], loudnesses_var[idx], loudnesses_interval[idx] = ut.get_loudness(track,h5)
    tempos[idx] = ut.get_tempo_feature(track,h5)
    time_signatures[idx] = ut.get_time_signature(track,h5)
    timbre_means[idx],timbre_vars[idx], timbre_median[idx], timbre_min[idx], timbre_max[idx] = ut.get_timbre(track,h5) 
    pitches_means[idx],pitches_vars[idx], pitches_median[idx], pitches_min[idx], pitches_max[idx]= ut.get_pitches(track,h5)
    energies[idx] = ut.get_energy_feature(track)
    h5.close()
    
#use binning for continious data
#problem: number of bins => freedman-driaconis rule
num_bins = 2* (stats.scoreatpercentile(loudnesses_interval, 75) - stats.scoreatpercentile(loudnesses_interval, 25))*len(loudnesses_interval)**(1/3)
bins = np.linspace(min(loudnesses_interval), max(loudnesses_interval),num=num_bins)
d_loudnesses_interval = np.digitize(loudnesses_interval, bins)

num_bins = 2* (stats.scoreatpercentile(loudnesses, 75) - stats.scoreatpercentile(loudnesses, 25))*len(loudnesses)**(1/3)
bins = np.linspace(min(loudnesses), max(loudnesses),num=100)
d_loudnesses = np.digitize(loudnesses, bins)

num_bins = 2* (stats.scoreatpercentile(tempos, 75) - stats.scoreatpercentile(tempos, 25))*len(tempos)**(1/3)
bins = np.linspace(min(tempos), max(tempos),num=100)
d_tempos = np.digitize(tempos, bins)

num_bins = 2* (stats.scoreatpercentile(energies, 75) - stats.scoreatpercentile(energies, 25))*len(energies)**(1/3)
bins = np.linspace(min(energies), max(energies),num=100)
d_energies = np.digitize(energies, bins)

d_timbre_means = np.empty((12),dtype='object')
d_timbre_vars = np.empty((12),dtype='object')
d_timbre_median = np.empty((12),dtype='object')
d_timbre_min = np.empty((12),dtype='object')
d_timbre_max = np.empty((12),dtype='object')
d_pitches_means = np.empty((12),dtype='object')
d_pitches_vars = np.empty((12),dtype='object')
d_pitches_median = np.empty((12),dtype='object')
d_pitches_min = np.empty((12),dtype='object')
d_pitches_max = np.empty((12),dtype='object')
for i in range(12):
    num_bins = 2* (stats.scoreatpercentile(timbre_means[:,i], 75) - stats.scoreatpercentile(timbre_means[:,i], 25))*len(timbre_means[:,i])**(1/3)
    bins = np.linspace(min(timbre_means[:,i]), max(timbre_means[:,i]),num=100)
    d_timbre_means[i] = np.digitize(timbre_means[:,i], bins)
    
    num_bins = 2* (stats.scoreatpercentile(timbre_vars[:,i], 75) - stats.scoreatpercentile(timbre_vars[:,i], 25))*len(timbre_vars[:,i])**(1/3)
    bins = np.linspace(min(timbre_vars[:,i]), max(timbre_vars[:,i]),num=100)
    d_timbre_vars[i] = np.digitize(timbre_vars[:,i], bins)

    num_bins = 2* (stats.scoreatpercentile(timbre_median[:,i], 75) - stats.scoreatpercentile(timbre_median[:,i], 25))*len(timbre_median[:,i])**(1/3)
    bins = np.linspace(min(timbre_median[:,i]), max(timbre_median[:,i]),num=100)
    d_timbre_median[i] = np.digitize(timbre_median[:,i], bins)
    
    num_bins = 2* (stats.scoreatpercentile(timbre_min[:,i], 75) - stats.scoreatpercentile(timbre_min[:,i], 25))*len(timbre_min[:,i])**(1/3)
    bins = np.linspace(min(timbre_min[:,i]), max(timbre_min[:,i]),num=100)
    d_timbre_min[i] = np.digitize(timbre_vars[:,i], bins)
    
    num_bins = 2* (stats.scoreatpercentile(timbre_max[:,i], 75) - stats.scoreatpercentile(timbre_max[:,i], 25))*len(timbre_max[:,i])**(1/3)
    bins = np.linspace(min(timbre_max[:,i]), max(timbre_max[:,i]),num=100)
    d_timbre_max[i] = np.digitize(timbre_max[:,i], bins)
    
    num_bins = 2* (stats.scoreatpercentile(pitches_means[:,i], 75) - stats.scoreatpercentile(pitches_means[:,i], 25))*len(pitches_means[:,i])**(1/3)
    bins = np.linspace(min(pitches_means[:,i]), max(pitches_means[:,i]),num=100)
    d_pitches_means[i] = np.digitize(pitches_means[:,i], bins)

    num_bins = 2* (stats.scoreatpercentile(pitches_vars[:,i], 75) - stats.scoreatpercentile(pitches_vars[:,i], 25))*len(pitches_vars[:,i])**(1/3)
    bins = np.linspace(min(pitches_vars[:,i]), max(pitches_vars[:,i]),num=100)
    d_pitches_vars[i] = np.digitize(pitches_vars[:,i], bins)

    num_bins = 2* (stats.scoreatpercentile(pitches_median[:,i], 75) - stats.scoreatpercentile(pitches_median[:,i], 25))*len(pitches_median[:,i])**(1/3)
    bins = np.linspace(min(pitches_median[:,i]), max(pitches_median[:,i]),num=100)
    d_pitches_median[i] = np.digitize(pitches_median[:,i], bins)

    num_bins = 2* (stats.scoreatpercentile(pitches_min[:,i], 75) - stats.scoreatpercentile(pitches_min[:,i], 25))*len(pitches_min[:,i])**(1/3)
    bins = np.linspace(min(pitches_min[:,i]), max(pitches_min[:,i]),num=100)
    d_pitches_min[i] = np.digitize(pitches_min[:,i], bins)

    num_bins = 2* (stats.scoreatpercentile(pitches_max[:,i], 75) - stats.scoreatpercentile(pitches_max[:,i], 25))*len(pitches_max[:,i])**(1/3)
    bins = np.linspace(min(pitches_max[:,i]), max(pitches_max[:,i]),num=100)
    d_pitches_max[i] = np.digitize(pitches_max[:,i], bins)

#use mutual information as a metric
mutuals = list([metrics.mutual_info_score(labels, modes),metrics.mutual_info_score(labels, keys),metrics.mutual_info_score(labels, d_loudnesses_interval),metrics.mutual_info_score(labels, d_loudnesses),metrics.mutual_info_score(labels, d_tempos),metrics.mutual_info_score(labels, time_signatures),metrics.mutual_info_score(labels, d_energies)])
for i in range(12):
    mutuals.append(metrics.mutual_info_score(labels, d_timbre_means[i]))
for i in range(12):
    mutuals.append(metrics.mutual_info_score(labels, d_pitches_means[i]))

print "mutual info for mode feature: %.2f" % mutuals[0]
print "mutual info for key feature: %.2f" % mutuals[1]
print "mutual info for loudness_interval feature: %.2f" % mutuals[2]
print "mutual info for loudness_mean feature: %.2f" % mutuals[3]
print "mutual info for tempo feature: %.2f" % mutuals[4]
print "mutual info for time_signature feature: %.2f" % mutuals[5]
print "mutual info for energy feature: %.2f" % mutuals[6]
for i in range(12):
    print "mutual info for timbre_means,timbre_vars, timbre_med, timbre_min, timbre_max %s feature: %.2f %.2f %.2f %.2f %.2f" % (str(i),metrics.mutual_info_score(labels, d_timbre_means[i]),metrics.mutual_info_score(labels, d_timbre_vars[i]),metrics.mutual_info_score(labels, d_timbre_median[i]),metrics.mutual_info_score(labels, d_timbre_min[i]),metrics.mutual_info_score(labels, d_timbre_max[i]))
for i in range(12):
    print "mutual info for pitch_means, pitch_vars, pitch_med, pitch_min, pitch_max %s feature: %.2f %.2f %.2f %.2f %.2f" % (str(i),metrics.mutual_info_score(labels, d_pitches_means[i]),metrics.mutual_info_score(labels, d_pitches_vars[i]),metrics.mutual_info_score(labels, d_pitches_median[i]),metrics.mutual_info_score(labels, d_pitches_min[i]),metrics.mutual_info_score(labels, d_pitches_max[i]))
print 'done'