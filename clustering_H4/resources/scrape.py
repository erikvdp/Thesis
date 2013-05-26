import urllib
import time
import random
import os
import sys
import platform

path = '/mnt/storage/data/msd/'
mean = 30.
def wait():
	seconds = 25. + random.expovariate(1. / mean)
	print 'going to sleep (s): ' + str(seconds)
	time.sleep(seconds)

f = open(path + '7digital_ids.txt')
song_ids = f.read().split('\n')

node = platform.node()
## node = 'clsnn001'
id = int(node[-2:]) - 1
num_nodes = 30

def print_and_log(text):
	print text
	with open(path + 'error_log__' + str(id) + '.txt', 'a') as log_file:
		log_file.write('\n' + text)

# id = int(sys.argv[1])
# num_nodes = int(sys.argv[2])

print_and_log("I'm the " + str(id) + "'th scraper of " + str(num_nodes))
print_and_log("(0-based)")

t = random.uniform(1, 200)
print 'going to sleep (s): ' + str(t)
time.sleep(t)
for i, song_id in enumerate(song_ids):
	song_id = song_id.strip()
	if i % num_nodes == id:
		new_path = path + "songs/" + song_id[0] + '/' + song_id[1] + '/'
		if not os.path.isfile(new_path + str(song_id) + ".clip.mp3"):
			print_and_log('going to download: ' + str(song_id))
			try:
				# download song:
				urllib.urlretrieve ("http://previews.7digital.com/clips/34/" + str(song_id) + ".clip.mp3", new_path + str(song_id) + ".clip.mp3")
			except:
				# error
				print_and_log('error: ' + str(sys.exc_info()[0]))
			# wait randomly anyhow
			wait()
		else:
			print_and_log('song exists: ' + str(song_id))