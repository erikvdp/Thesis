import cPickle as pickle
import hdf5_getters as GETTERS


DATAFILE = "../msd_dense_subset/dense/songs_tracks.pkl"  #mood tracks for subset
IDFILE = "ids.txt"

with open(IDFILE, 'w') as f1, open(DATAFILE, 'r') as f2:
        data = pickle.load(f2)
        for track in data.values():
            path = "../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
            h5 = GETTERS.open_h5_file_read(path)
            digitalid = str(GETTERS.get_track_7digitalid(h5))
            h5.close()
            f1.write(digitalid+'\n')
print "done"
