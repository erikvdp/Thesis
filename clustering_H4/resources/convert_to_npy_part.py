"""

script to convert HDF5 data to NPY

"""


import numpy as np
import sys
import os
import h5py
import shutil


if len(sys.argv) != 2:
    print "Usage: convert_to_npy_part.py <part_number>"
    sys.exit(1)

SOURCE_PATH_MASK = "/mnt/storage/data/msd/dataset_raw/dataset_raw_%04d.h5"
TARGET_FILENAME_MASK = "dataset_raw_%04d_%s.npy"
TARGET_PATH = "/mnt/storage/data/msd/dataset_raw_npy/"
TMP_PATH = "/tmp"


# figure out which part of the data to process
part_number = int(sys.argv[1])

print "Convert HDF5 to npy part %d" % part_number
print


source_path = SOURCE_PATH_MASK % part_number
print "Open raw data file %s" % source_path
source_file = h5py.File(source_path, 'r')
X = source_file['X'][()]
Y = source_file['Y'][()]



target_filename_X = TARGET_FILENAME_MASK % (part_number, 'X')
target_path_X = os.path.join(TMP_PATH, target_filename_X)
final_path_X = os.path.join(TARGET_PATH, target_filename_X)
print "Save to temporary target file %s" % target_path_X
np.save(target_path_X, X)

target_filename_Y = TARGET_FILENAME_MASK % (part_number, 'Y')
target_path_Y = os.path.join(TMP_PATH, target_filename_Y)
final_path_Y = os.path.join(TARGET_PATH, target_filename_Y)
print "Save to temporary target file %s" % target_path_Y
np.save(target_path_Y, Y)


print "Copying %s to %s..." % (target_path_X, final_path_X)
shutil.copyfile(target_path_X, final_path_X)

print "Copying %s to %s..." % (target_path_Y, final_path_Y)
shutil.copyfile(target_path_Y, final_path_Y)

print "Removing %s..." % target_path_X
os.remove(target_path_X)

print "Removing %s..." % target_path_Y
os.remove(target_path_Y)

print "Done."
