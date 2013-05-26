"""

script to build part of the dataset and store it in an HDF5 file.

"""
import numpy as np
import sys
import os
import subprocess
import cPickle as pickle
import h5py
from scipy.io import wavfile
import shutil


if len(sys.argv) != 2:
    print "Usage: build_dataset_part.py <part_number>"
    sys.exit(1)

INT_CLIP_PATH_MAP_PATH = "/mnt/storage/data/msd/int_clip_path_map.pkl"
FACTORS_PATH = "/mnt/storage/data/msd/factors_real_for_audiorec.pkl"
CLIPS_PATH = "/mnt/storage/data/msd/extracted_clips"
TARGET_FILENAME_MASK = "dataset_raw_%04d.h5"
TARGET_PATH = "/mnt/storage/data/msd/dataset_raw/"
TMP_PATH = "/tmp"


NUM_SONGS = 385371 # the total number of songs for which we have factors
NUM_PARTS = 200 # number of files to spread the data over
CLIP_SAMPLE_LENGTH = 29 * 22050 # clip length in samples, this is exact


# load necessary mappings / data
print "Load int id to clip path mapping..."
with open(INT_CLIP_PATH_MAP_PATH, 'r') as f:
    int_clip_path_map = pickle.load(f)
    
print "Load factors..."
with open(FACTORS_PATH, 'r') as f:
    d = pickle.load(f)
factors = d['V'] # only the item factors
num_factors = factors.shape[1] # number of latent factors
    

# figure out which part of the data to process
part_number = int(sys.argv[1])
num_ids_per_part = int(np.ceil(NUM_SONGS / float(NUM_PARTS)))

start = part_number * num_ids_per_part
end = (part_number + 1) * num_ids_per_part

print "Build dataset part %d: IDs %d to %d" % (part_number, start, end - 1)
print


# figure out how many songs have a clip path
num_songs_with_clip = 0
for int_id in xrange(start, end):
    if int_id in int_clip_path_map:
        num_songs_with_clip += 1
        
print "Gathering %d clips (int_id range length: %d)" % (num_songs_with_clip, end - start)


# now create the HDF5 file and arrays
target_filename = TARGET_FILENAME_MASK % part_number
target_path = os.path.join(TMP_PATH, target_filename)
final_path = os.path.join(TARGET_PATH, target_filename)
print "Creating target file %s..." % target_path
target_file = h5py.File(target_path, 'w')

X = target_file.create_dataset('X', (num_songs_with_clip, CLIP_SAMPLE_LENGTH), 'int16', compression='lzf')
Y = target_file.create_dataset('Y', (num_songs_with_clip, num_factors), 'float32', compression='lzf')

idx = 0
for int_id in xrange(start, end):
    progress = (int_id - start) * 100 / float(end - start)
    if int_id not in int_clip_path_map:
        print "Skipping %d, no clip path given" % int_id
        continue
        
    wav_path = int_clip_path_map[int_id]
    print "(%.2f%%) Processing %d: %s" % (progress, int_id, wav_path)
    _, wav_data = wavfile.read(wav_path)
    X[idx] = wav_data
    Y[idx] = factors[int_id]
    idx += 1

print "Closing target file..."
target_file.close()

print "Copying %s to %s..." % (target_path, final_path)
shutil.copyfile(target_path, final_path)

print "Removing %s..." % target_path
os.remove(target_path)

print "Done."
