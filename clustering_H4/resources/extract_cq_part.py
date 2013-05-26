"""

script to extract a constant-q / pitch representation of a raw dataset part.

IMPORTANT: no loudness logarithm is taken yet, so the constant can be tuned afterwards.

"""


import numpy as np
import sys
import os
import h5py
import shutil

import cq

if len(sys.argv) != 2:
    print "Usage: extract_cq_part.py <part_number>"
    sys.exit(1)

SOURCE_PATH_MASK = "/mnt/storage/data/msd/dataset_raw/dataset_raw_%04d.h5"
TARGET_FILENAME_MASK = "dataset_cq_%04d.h5"
TARGET_PATH = "/mnt/storage/data/msd/dataset_cq/"
TMP_PATH = "/tmp"

SAMPLE_FREQ = 22050
WINDOW_SIZE = 1024
WINDOW_SHIFT = WINDOW_SIZE / 2
CQ_LEN = 1249 # this is hardcoded which is a bit nasty... but I can't be arsed to
# figure out why it is 1249 and not 1247 for the default parameters, like it would
# be with the mel spectrograms.

# figure out which part of the data to process
part_number = int(sys.argv[1])

print "Extract constant Q representation part %d" % part_number
print

print "Compute filter bank"
# compute the filters to extract the constant Q features with
filter_bank = cq.pitch_filter_bank(fs=SAMPLE_FREQ)


source_path = SOURCE_PATH_MASK % part_number
print "Open raw data file %s" % source_path
source_file = h5py.File(source_path, 'r')
X = source_file['X']

num_clips, clip_len = X.shape
print "  %d clips, %d samples per clip" % (num_clips, clip_len)


print "%d pitch windows per clip" % CQ_LEN
num_pitches = cq.num_pitches


target_filename = TARGET_FILENAME_MASK % part_number
target_path = os.path.join(TMP_PATH, target_filename)
final_path = os.path.join(TARGET_PATH, target_filename)
print "Open temporary target file %s" % target_path
target_file = h5py.File(target_path, 'w')
Xt = target_file.create_dataset('X', (num_clips, CQ_LEN, num_pitches), 'float32', compression='lzf')

print "Copy over the labels"
source_file.copy('Y', target_file)


for k in xrange(num_clips):
    pct = k * 100 / float(num_clips)
    print "(%.2f%%) clip %d of %d" % (pct, k, num_clips)
    
    audio = X[k] / float(2**15) # normalise so the samples are between -1 and +1
    pitch_rep = cq.audio_to_pitch_array(audio, filter_bank, window_size=WINDOW_SIZE, window_shift=WINDOW_SHIFT).T.astype('float32')
    Xt[k] = pitch_rep


print "Closing target file..."
target_file.close()

print "Copying %s to %s..." % (target_path, final_path)
shutil.copyfile(target_path, final_path)

print "Removing %s..." % target_path
os.remove(target_path)

print "Done."
