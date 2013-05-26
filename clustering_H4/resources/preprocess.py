"""

Script that preprocesses a single clip file. 
    
"""

import sys
import os
import subprocess

max_length = 29 # maximal clip length in seconds

if len(sys.argv) != 3:
    print "Usage: preprocess.py <clip_file> <output_file>"
    sys.exit(1)
    
src_path = sys.argv[1]
tgt_path = sys.argv[2]

fn = os.path.basename(src_path).replace(".mp3", ".tmp.wav")
tmp_path = os.path.join("/tmp/", fn)

print ">> convert MP3 file to WAV"
command = "lame --silent --decode %s %s" % (src_path, tmp_path)
print command
os.system(command)

print ">> get clip length"
s = subprocess.check_output('soxi -D %s' % tmp_path, shell=True)
clip_length = float(s.strip())
print "  length: %.2f seconds" % clip_length
trim_start = max((clip_length - max_length) / 2, 0)
print "  cutting %.2f seconds starting from %.2f seconds" % (max_length, trim_start)

print ">> process WAV file with sox"
command = "sox -v 0.9 \"%s\" \"%s\" trim %.4f %.4f rate 22050 remix -" % (tmp_path, tgt_path, trim_start, max_length)
# converts to mono (remix -), downsamples to 22kHz (rate 22050) and attenuates a little to avoid clipping (-v 0.9)
# cuts 29 seconds from the file (trim <start> <duration>). if it's shorter, the file will be kept (we will deal with this later)
print command
os.system(command)

print ">> remove temporary WAV file"
os.remove(tmp_path)



