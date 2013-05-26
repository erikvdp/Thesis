# mel scale related stuff.

import numpy as np


def freq2mel(freq):
    return 1127.01048 * np.log(1 + freq / 700.0)

def mel2freq(mel):
    return (np.exp(mel / 1127.01048) - 1) * 700

def mel_binning_matrix(specgram_window_size, sample_frequency, num_mel_bands):
    """
    function that returns a matrix that converts a regular DFT to a mel-spaced DFT,
    by binning coefficients.
    
    specgram_window_size: the window length used to compute the spectrograms
    sample_frequency: the sample frequency of the input audio
    num_mel_bands: the number of desired mel bands.
    
    The output is a matrix with dimensions (specgram_window_size/2 + 1, num_bands)
    """
    min_freq, max_freq = 0, sample_frequency / 2
    min_mel = freq2mel(min_freq)
    max_mel = freq2mel(max_freq)
    num_specgram_components = specgram_window_size / 2 + 1
    m = np.zeros((num_specgram_components, num_mel_bands))
    
    r = np.arange(num_mel_bands + 2) # there are (num_mel_bands + 2) filter boundaries / centers

    # evenly spaced filter boundaries in the mel domain:
    mel_filter_boundaries = r * (max_mel - min_mel) / (num_mel_bands + 1) + min_mel
    
    def coeff(idx, mel): # gets the unnormalised filter coefficient of filter 'idx' for a given mel value.
        lo, cen, hi = mel_filter_boundaries[idx:idx+3]
        if mel <= lo or mel >= hi:
            return 0
        # linearly interpolate
        if lo <= mel <= cen:
            return (mel - lo) / (cen - lo)
        elif cen <= mel <= hi:
            return 1 - (mel - cen) / (hi - cen)
            
    
    for k in xrange(num_specgram_components):
        # compute mel representation of the given specgram component idx
        freq = k / float(num_specgram_components) * (sample_frequency / 2)
        mel = freq2mel(freq)
        for i in xrange(num_mel_bands):
            m[k, i] = coeff(i, mel)

    # normalise so that each filter has unit contribution
    return m / m.sum(0)
    
    
    
def loudness_log(data, C=10.0**5):
    return np.log(1 + data * C) # convert to a logarithmic loudness scale
