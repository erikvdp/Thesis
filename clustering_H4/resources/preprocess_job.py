import numpy as np
import linecache # random access to text lines
import os
import sys

SRC_CLIPS_PATH = "./songs"
TGT_CLIPS_PATH = "./extracted_clips"
IDS_FILE = "problemids.txt"
PREPROCESS_SCRIPT_PATH = "preprocess.py"
NUM_TRACKS_TOTAL = 37
NUM_JOBS = 1 # number of jobs to divide the task over

num_tracks_per_job = int(np.ceil(NUM_TRACKS_TOTAL / float(NUM_JOBS)))


job_id = int(sys.argv[1]) # should be 0-based!

print "I am job %d" % job_id
print

start = min(job_id * num_tracks_per_job, NUM_TRACKS_TOTAL)
end = min((job_id + 1) * num_tracks_per_job, NUM_TRACKS_TOTAL)

line_numbers = xrange(start + 1, end + 1) # line numbers are 1-based

for ln in line_numbers:
    line = linecache.getline(IDS_FILE, ln).strip()
    if line == '':
        continue # the line does not exist in the file.
        
    track_id = line # keep it as a string, we don't need it as int
    
    # find the clip file that matches
    src_clip_subpath = "%s.clip.mp3" % (track_id)
    src_clip_path = os.path.join(SRC_CLIPS_PATH, src_clip_subpath)
    tgt_clip_subpath = "%s.clip.wav" % (track_id)
    tgt_clip_path = os.path.join(TGT_CLIPS_PATH, tgt_clip_subpath)
    
    if not os.path.exists(src_clip_path):
        print "%d. %s: clip does not exist, skipping." % (ln, track_id)
        continue
        
    if os.path.exists(tgt_clip_path):
        print "%d. %s: clip has already been processed (target exists), skipping." % (ln, track_id)
        continue
        
    print "%d. %s: processing." % (ln, track_id)
    command = "python %s %s %s" % (PREPROCESS_SCRIPT_PATH, src_clip_path, tgt_clip_path)
    # print "  >> %s" % command
    os.system(command)
    
    
    
    
    
