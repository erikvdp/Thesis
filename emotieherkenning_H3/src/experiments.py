'''
Created on 29 nov. 2012

@author: Erik Vandeputte
'''
import util as ut
import echonestparser as parser
import echonestanalyzer as analyzer

import lyricsparser as l_parser
import lyricsanalyzer as l_analyzer

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import sklearn.svm as svm
import sklearn.neighbors as neighbors



SOURCE_DATA_FILE = "../../msd_dense_subset/mood.txt"
SOURCE_DATA_FILE_2 = "../../msd_dense_subset/mood2.txt"

E_TARGET_DATA_FILE = "../../msd_dense_subset/mood_echonest_features.pkl"
E_TARGET_DATA_FILE_2 = "../../msd_dense_subset/mood_echonest_features_2.pkl"

L_TARGET_DATA_FILE = "../../msd_dense_subset/mood_lyrics_features.pkl"
L_TARGET_DATA_FILE_2 = "../../msd_dense_subset/mood_lyrics_features2.pkl"



def experiment_1():
    ut.write_mood(2)
    ut.write_mood_2()
    
    parser.parse_file(SOURCE_DATA_FILE,E_TARGET_DATA_FILE)
    parser.parse_file(SOURCE_DATA_FILE_2,E_TARGET_DATA_FILE_2)
    
    #hyperparameters were obtained using gridsearch (see def optimize_par(data_file) in echonestanalyzer.py)
    classifiers_1 = [neighbors.KNeighborsClassifier(6, weights='distance'), svm.SVC(kernel='poly', C=100, degree=3),svm.SVC(kernel='rbf', C=1, gamma = 0.0001),RandomForestClassifier(n_estimators=400)]
    classifiers_2 = [neighbors.KNeighborsClassifier(8, weights='distance'), svm.SVC(kernel='poly', C=10, degree=1),svm.SVC(kernel='rbf', C=10, gamma = 0.001),RandomForestClassifier(n_estimators=500)]
    for classifier in classifiers_1:
        analyzer.cross_val(E_TARGET_DATA_FILE, classifier)
    for classifier in classifiers_2:
        analyzer.cross_val(E_TARGET_DATA_FILE_2, classifier)

def experiment_2():
    #this experiment is only for dataset1 (contains anger mood tracks)
    classifier = RandomForestClassifier(n_estimators=400)
    
    #2 classes
    ut.write_mood(2)
    parser.parse_file(SOURCE_DATA_FILE,E_TARGET_DATA_FILE)
    analyzer.cross_val(E_TARGET_DATA_FILE, classifier)
    analyzer.run_analyzer(E_TARGET_DATA_FILE, classifier)
    
    
    #3 classes
    #dataset1
    ut.write_mood(3)
    #parse features
    parser.parse_file(SOURCE_DATA_FILE,E_TARGET_DATA_FILE)
    #run evaluation
    analyzer.cross_val(E_TARGET_DATA_FILE, classifier)
    #run single evaluation and return the classificaton errors
    analyzer.run_analyzer(E_TARGET_DATA_FILE, classifier)

def experiment_3():
    classifier_1  = svm.LinearSVC()
    classifier_2 = MultinomialNB()
    
    #SENTIMENT WORDS
    #dataset 1
    l_parser.parse_file(SOURCE_DATA_FILE,L_TARGET_DATA_FILE , 1)
    #dataset 2
    l_parser.parse_file(SOURCE_DATA_FILE_2,L_TARGET_DATA_FILE_2 , 1)
    
    
    l_analyzer.cross_val(L_TARGET_DATA_FILE,classifier_1)
    l_analyzer.cross_val(L_TARGET_DATA_FILE,classifier_2)
    
    l_analyzer.cross_val(L_TARGET_DATA_FILE_2,classifier_1)
    l_analyzer.cross_val(L_TARGET_DATA_FILE_2,classifier_2)
    
    #ALL WORDS
    
    l_parser.parse_file(SOURCE_DATA_FILE,L_TARGET_DATA_FILE , 2)
    l_parser.parse_file(SOURCE_DATA_FILE_2,L_TARGET_DATA_FILE_2 , 2)
    
    l_analyzer.cross_val(L_TARGET_DATA_FILE,classifier_1)
    l_analyzer.cross_val(L_TARGET_DATA_FILE,classifier_2) 
    
    l_analyzer.cross_val(L_TARGET_DATA_FILE_2,classifier_1)
    l_analyzer.cross_val(L_TARGET_DATA_FILE_2,classifier_2)
    
experiment_2()