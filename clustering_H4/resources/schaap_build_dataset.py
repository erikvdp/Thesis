"""

Script to build the entire dataset on schaap

"""

import os

num_parts = 200

for k in xrange(num_parts):
    print ">>> BUILDING PART %d" % k
    print 
    command = "epython build_dataset_part.py %d" % k
    os.system(command)
