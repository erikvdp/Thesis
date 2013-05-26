'''
Created on 17 nov. 2012

@author: Erik Vandeputte
'''
import numpy as np
import cPickle as pickle

LYRICS_FILE = "../../msd_dense_subset/mxm/mxm_dataset.txt" #contains all lyrics info

SENTIMENT_WORDS_FILE1 = "../../msd_dense_subset/mxm/AFINN/AFINN-111.txt"
SENTIMENT_WORDS_FILE2 = "../../msd_dense_subset/mxm/GI"
SENTIMENTS_WORDS_MCDONALD_POS = "../../msd_dense_subset/mxm/LoughranMcDonald_Positive.txt"
SENTIMENTS_WORDS_MCDONALD_NEG = "../../msd_dense_subset/mxm/LoughranMcDonald_Negative.txt"

TARGET_FILE = '../../msd_dense_subset/sentiment_words.pkl' 

def load_sentiment_words(popular_words):
    with open(SENTIMENT_WORDS_FILE1, 'r') as f:
        idx = list()
        for line in f:
            word, _ = line.strip().split('\t')
            if word in popular_words:
                idx.append(popular_words.index(word))
    with open(SENTIMENT_WORDS_FILE2, 'r') as f:
        for line in f:
            word = line.strip()
            if word.lower() in popular_words:
                idx.append(popular_words.index(word.lower()))
    return set(idx)

def load_sentiment_words_McDonald(popular_words):
    #deprecated
    with open(SENTIMENTS_WORDS_MCDONALD_NEG, 'r') as f:
        idx = list()
        for line in f:
            word, _ = line.strip().split(',')
            if word.lower()in popular_words:
                idx.append(popular_words.index(word.lower()))
    with open(SENTIMENTS_WORDS_MCDONALD_POS, 'r') as f:
        for line in f:
            word, _ = line.strip().split(',')
            if word.lower()in popular_words:
                idx.append(popular_words.index(word.lower()))
    return set(idx)


with open(LYRICS_FILE, 'r') as lyricsfile:
        line = lyricsfile.readline()
        while(line.startswith( '#' )):
            line = lyricsfile.readline(); #skip first lines
        words = line.strip().split(',')
        words[0] = "i" #remove the % sign
        #popular_words_idx = load_sentiment_words_McDonald(words)
        popular_words_idx = load_sentiment_words(words)
        words = np.array(words)
        popular_words = words[list(popular_words_idx)]
        print popular_words
        print len(popular_words)
        
data = {
        'words': popular_words,
        'indices' : list(popular_words_idx)}
with open(TARGET_FILE, 'w') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
print "data saved to %s" %TARGET_FILE