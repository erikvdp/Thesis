"""

script to map integer ids (positions in the factor matrix) to clip paths.
This requires an int->clip map to be available already.

"""
import os
import subprocess
import cPickle as pickle



INT_CLIP_MAP_PATH = "/mnt/storage/data/msd/int_clip_map.pkl"
CLIPS_PATH = "/mnt/storage/data/msd/extracted_clips"

TARGET_PATH = "/mnt/storage/data/msd/int_clip_path_map.pkl"

MIN_CLIP_LENGTH = 29.0 # minimum length of the clips in seconds.


# load necessary mappings
print "load int id to clip id mapping..."
with open(INT_CLIP_MAP_PATH, 'r') as f:
    int_clip_map = pickle.load(f)

num_songs = len(int_clip_map)

clip_path_map = {}
skipped_no_clip_file = 0
skipped_not_long_enough = 0

for k, (int_id, seven_id) in enumerate(int_clip_map.iteritems()): 
    # find out if a clip exists
    clip_subpath = "%s/%s/%s.clip.wav" % (seven_id[0], seven_id[1], seven_id)
    clip_path = os.path.join(CLIPS_PATH, clip_subpath)
    
    if not os.path.exists(clip_path):
        print "clip %s does not exist, skipping." % seven_id
        skipped_no_clip_file += 1
        continue
    
    # find out if the clip is long enough
    s = subprocess.check_output('soxi -D %s' % clip_path, shell=True)
    clip_length = float(s.strip())
    
    if clip_length < MIN_CLIP_LENGTH:
        print "clip %s is not long enough (only %.2f s, should be %.2f s)." % (seven_id, clip_length, MIN_CLIP_LENGTH)
        skipped_not_long_enough += 1
        continue
    
    # keep this mapping
    clip_path_map[int_id] = clip_path
    
    if k % 1000 == 0:
        print "%.1f %%" % (k * 100 / float(num_songs))
    
    
print "%d missing clip files" % skipped_no_clip_file
print "%d clips not long enough" % skipped_not_long_enough
print

with open(TARGET_PATH, 'w') as f:
    pickle.dump(clip_path_map, f, pickle.HIGHEST_PROTOCOL)
    
print "stored in %s." % TARGET_PATH
