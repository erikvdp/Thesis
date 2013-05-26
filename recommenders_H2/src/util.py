'''
Created on 9 okt. 2012

@author: Erik Vandeputte
'''
import random
import cPickle as pickle
import time
import hdf5_getters as GETTERS

TRAIN_TRIPLETS_FILE = '../../msd_dense_subset/train_triplets_dense_subset.txt'
VALIDATION_TRIPLETS_FILE = '../../msd_dense_subset/validation_triplets_dense_subset.txt'
DATA_TRIPLETS_FILE = '../../msd_dense_subset/data_dense_subset.txt'

#returns dictionary with number of plays of each song
def song_to_count(if_str):
	stc=dict()
	with open(if_str,"r") as f:
		for line in f:
			_,song,count = line.strip().split('\t')
			if song in stc:
				stc[song] +=int(count)
			else:
				stc[song] =int(count)
	return stc

#returns dictionary with number of plays of each user
def user_to_count(if_str):
	utc=dict()
	with open(if_str,"r") as f:
		for line in f:
			user,_,_ = line.strip().split('\t')
			if user in utc:
				utc[user] +=1
			else:
				utc[user] =1
	return utc
#returns the artist for this song
def song_to_artist(if_str):
	songs_tracks = pickle.load(open ("../../msd_dense_subset/dense/songs_tracks.pkl",'r'));
	track = str(songs_tracks[if_str])
	# build path
	path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
	h5 = GETTERS.open_h5_file_read(path)
	artist = GETTERS.get_artist_name(h5)
	h5.close()
	return artist
	

#returns info for a given song in subset data
#f.e. songinfo("SOCFPSZ12A6D4FCA89")
def songinfo(if_str):
	songs_tracks = pickle.load(open ("../../msd_dense_subset/dense/songs_tracks.pkl",'r'));
	track = str(songs_tracks[if_str])
	# build path
	path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
	h5 = GETTERS.open_h5_file_read(path)
	artist_name = GETTERS.get_artist_name(h5)
	song_name = GETTERS.get_title(h5)
	year = GETTERS.get_year(h5, 0)
	#segments = GETTERS.get_segments_start(h5, 0);
	#segments_pitches = GETTERS.get_segments_pitches(h5, 0)
	h5.close()
	return artist_name+ " - " +song_name + " (" +str(year) +")"

#returns some useful stats about a certain user
def userstats(if_str):
	songs = dict()
	with open(DATA_TRIPLETS_FILE,"r") as f:
		for line in f:
			user,song,count = line.strip().split('\t')
			if user == if_str:
				songs[song] = int(count)
	totalplays = sum(songs.values())
	s = sorted(songs, key=songs.get,reverse=True)
	#show some stats
	print "total listens: %s\n" % totalplays
	for i in range(10):
		print songinfo(s[i]) + "\t"+ str(songs[s[i]])
#makes a validation and test set
#loop over the data and add with a 20% chance to the validation set
#dirty hack = make sure each user and each song is added to validation set and training set
#TIME = 5.4 sec
def make_validation_set():
	start_time = time.time()
	validationfile = open(VALIDATION_TRIPLETS_FILE, 'w')
	trainfile = open(TRAIN_TRIPLETS_FILE, 'w')
	t_users = set()
	t_songs = set()
	with open(DATA_TRIPLETS_FILE, 'r') as f:
		for line in f:
			user,song,_ = line.strip().split('\t')
			if user not in t_users or song not in t_songs:
				t_users.add(user)
				t_songs.add(song)
				trainfile.write(line)
				continue
			elif(random.random() >= 0.8):
				validationfile.write(line)
			else:
				trainfile.write(line)
	validationfile.close()
	trainfile.close()
	print "  took %.2f seconds" % (time.time() - start_time)

#SCRIPT