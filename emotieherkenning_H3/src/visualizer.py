'''
Created on 10 nov. 2012

@author: Enrico
'''
import hdf5_getters as GETTERS
import util as ut
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def plot_loudness_curve(track):
    path = "../../msd_dense_subset/mood/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
    h5 = GETTERS.open_h5_file_read(path)
    segments = (GETTERS.get_segments_start(h5))
    sections = (GETTERS.get_sections_start(h5))
    max_loudness = (GETTERS.get_segments_loudness_max(h5))
    loudness = (GETTERS.get_segments_loudness_start(h5))
    average_loudness = (max_loudness + loudness) / 2
    average_loudness_song = GETTERS.get_loudness(h5)
    start_fade_out = GETTERS.get_start_of_fade_out(h5)
    end_fade_in = GETTERS.get_end_of_fade_in(h5)
    plt.title('loudness curve for ' + ut.get_track_info(track))
    plt.ylabel('Loudness(dB)')
    plt.xlabel('Time(s)')
    plt.plot(segments, average_loudness, label='Filtered')
    plt.axhline(average_loudness_song, color='green',label='average')
    for section in enumerate(sections):
        plt.axvline(section[1], color='red')
    plt.axvline(start_fade_out, color='green')
    plt.axvline(end_fade_in, color='green')
    #plt.legend(('average'))
    h5.close()
    plt.show()

def plot_timbre(track):   
    path = "../../msd_dense_subset/mood/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
    h5 = GETTERS.open_h5_file_read(path)
    timbres = GETTERS.get_segments_timbre(h5)
    segments = GETTERS.get_segments_start(h5)
    sections = GETTERS.get_sections_start(h5)
    idx = list()
    for section in sections:
        dif = segments - section
        posdif = np.where(dif >=0)
        idx.append(posdif[0][0])
    plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Chroma values for ' + ut.get_track_info(track))
    plt.ylabel('Chroma values')
    plt.xlabel('Time')
    plt.xticks(idx,sections.astype(int))
    extent =[0,segments.shape[0],0,12]
    plt.imshow(timbres.transpose(), extent = extent,aspect = 'auto',interpolation = 'nearest' ,origin='lower',vmin=-100, vmax=100)
    plt.colorbar()
    plt.autoscale(enable=True, axis='x')
    #plt.pcolor(pitches.transpose())
    h5.close()
    plt.show()


 
def plot_chroma(track):   
    path = "../../msd_dense_subset/mood/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
    h5 = GETTERS.open_h5_file_read(path)
    pitches = GETTERS.get_segments_pitches(h5)
    segments = GETTERS.get_segments_start(h5)
    sections = GETTERS.get_sections_start(h5)
    idx = list()
    for section in sections:
        dif = segments - section
        posdif = np.where(dif >=0)
        idx.append(posdif[0][0])
    plt.figure(num=None, figsize=(20, 12), dpi=80, facecolor='w', edgecolor='k')
    plt.title('Chroma values for ' + ut.get_track_info(track))
    plt.ylabel('Chroma values')
    plt.xlabel('Time')
    plt.xticks(idx,sections.astype(int))
    extent =[0,segments.shape[0],0,12]
    plt.imshow(pitches.transpose(), extent = extent,aspect = 'auto',interpolation='nearest',origin='lower')
    plt.colorbar()
    plt.autoscale(enable=True, axis='x')
    #plt.pcolor(pitches.transpose())
    h5.close()
    plt.show()

def plots(track):
    f, axarr = plt.subplots(2, sharex=True)
    path = "../../msd_dense_subset/dense/"+track[2]+"/"+track[3]+"/"+track[4]+"/"+track+".h5"
    h5 = GETTERS.open_h5_file_read(path)
    segments = (GETTERS.get_segments_start(h5))
    sections = (GETTERS.get_sections_start(h5))
    max_loudness = (GETTERS.get_segments_loudness_max(h5))
    loudness = (GETTERS.get_segments_loudness_start(h5))
    average_loudness = (max_loudness + loudness) / 2
    average_loudness_song = GETTERS.get_loudness(h5)
    start_fade_out = GETTERS.get_start_of_fade_out(h5)
    end_fade_in = GETTERS.get_end_of_fade_in(h5)
    pitches = GETTERS.get_segments_pitches(h5)
    h5.close()
    
    
    axarr[0].set_title('loudness curve for ' + ut.get_track_info(track))
    axarr[0].plot(segments, average_loudness, label='Filtered')
    axarr[0].axhline(average_loudness_song, color='green')
    for section in enumerate(sections):
        axarr[0].axvline(section[1], color='red')
    axarr[0].axvline(start_fade_out, color='green')
    axarr[0].axvline(end_fade_in, color='green')
    
    
    idx = list()
    for section in sections:
        dif = segments - section
        posdif = np.where(dif >=0)
        idx.append(posdif[0][0])
    axarr[1].set_title('Chroma values for ' + ut.get_track_info(track))
    #axarr[1].set_xticks(idx,sections.astype(int))
    extent =[0,segments.shape[0],0,12]
    axarr[1].imshow(pitches.transpose(),extent = extent,aspect = 'auto',interpolation='nearest',origin='lower')
    plt.show()
#SCRIPT
track = 'TRRABGI128E0780C8A' #coldplay - politik
track = 'TRTTZDE128C71969A1' #U2- beautiful day (set to mood dataset)
#plots(track)
plot_loudness_curve(track)
plot_chroma(track)
#plot_timbre(track)