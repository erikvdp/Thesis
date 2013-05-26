"""

script to extract a mel spectrogram representation of a raw dataset part.

"""

import numpy as np
import sys
import os
import h5py
import shutil

import mel
from matplotlib.mlab import specgram

if len(sys.argv) != 2:
    print "Usage: extract_melspec_part.py <part_number>"
    sys.exit(1)

SOURCE_PATH_MASK = "/mnt/storage/data/msd/dataset_raw/dataset_raw_%04d.h5"
TARGET_FILENAME_MASK = "dataset_mel_%04d.h5"
TARGET_PATH = "/mnt/storage/data/msd/dataset_mel/"
TMP_PATH = "/tmp"

DFT_SIZE = 1024
DFT_OVERLAP = DFT_SIZE / 2
NUM_MEL_BANDS = 128
SAMPLE_FREQ = 22050



# figure out which part of the data to process
part_number = int(sys.argv[1])

print "Extract mel spectrograms part %d" % part_number
print

print "Compute mel binning matrix"
# compute the matrix for converting a spectrogram to a mel-spectrogram
mel_matrix = mel.mel_binning_matrix(DFT_SIZE, SAMPLE_FREQ, NUM_MEL_BANDS).astype('float32')


source_path = SOURCE_PATH_MASK % part_number
print "Open raw data file %s" % source_path
source_file = h5py.File(source_path, 'r')
X = source_file['X']

num_clips, clip_len = X.shape
print "  %d clips, %d samples per clip" % (num_clips, clip_len)

# compute length of the melspectrogram of a sample
spec_len = 1 + (clip_len - DFT_SIZE) / (DFT_SIZE - DFT_OVERLAP) # number of spectrogram windows
print "  => %d spectrogram windows per clip" % spec_len


target_filename = TARGET_FILENAME_MASK % part_number
target_path = os.path.join(TMP_PATH, target_filename)
final_path = os.path.join(TARGET_PATH, target_filename)
print "Open temporary target file %s" % target_path
target_file = h5py.File(target_path, 'w')
Xt = target_file.create_dataset('X', (num_clips, spec_len, NUM_MEL_BANDS), 'float32', compression='lzf')

print "Copy over the labels"
source_file.copy('Y', target_file)


for k in xrange(num_clips):
    pct = k * 100 / float(num_clips)
    print "(%.2f%%) clip %d of %d" % (pct, k, num_clips)
    
    audio = X[k] / float(2**15) # normalise so the samples are between -1 and +1
    s, freqs, t = specgram(audio, NFFT=DFT_SIZE, noverlap=DFT_OVERLAP)
    ss = s.T.astype('float32')
    ss_mel = np.dot(ss, mel_matrix)
    st = mel.loudness_log(ss_mel)
    Xt[k] = st
    

print "Closing target file..."
target_file.close()

print "Copying %s to %s..." % (target_path, final_path)
shutil.copyfile(target_path, final_path)

print "Removing %s..." % target_path
os.remove(target_path)

print "Done."
