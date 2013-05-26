"""
Tools for building / working with constant Q representations (pitch representations)
most of this is scavenged from sequences/magnatagatune.py
"""

import numpy as np
import scipy.signal as sig


def pitch2freq(p):
    return 440 * 2 ** ((p - 69)/12.0)
    
def freq2pitch(f):
    return (np.log(f / 440.0) / np.log(2)) * 12 + 69
    
def center_frequencies(lowest=-57, highest=50):
    return 440 * 2 ** (np.arange(lowest, highest + 1) / 12.0)


RATIOS_PITCHES_DEFAULT = [(1, range(83, 115 + 1)), (4, range(47, 82 + 1)), (16, range(16, 46 + 1))]
# technically, 118 is the highest possible pitch, but the coefficients start to behave strangely.

num_pitches = sum([len(v[1]) for v in RATIOS_PITCHES_DEFAULT])

def pitch_filter_bank(ratios_pitches=None, fs=16000, Q=25.0, max_loss_pass=1.0, min_attenuation_stop=50.0):
    """
    lowest pitch: 20.6 Hz = pitch 16, the lowest pitch above the low threshold of hearing
    highest pitch: 7458.6 Hz = pitch 118, the highest pitch below half of the sampling frequency (fs = 16000Hz)
    Note that 119 is technically below the nyquist frequency (~7900Hz), but the right stopband frequency wouldn't be.
    
    fs: sampling frequency of the input in Hz
    Q: Q factor = frequency / bandwidth, used to determine passband and stopband frequencies of the elliptic filters
    max_loss_pass: maximal loss in passband in dB
    min_attenuation_stop: minimal attenuation in stopband in dB
    """
    if ratios_pitches is None:
        ratios_pitches = RATIOS_PITCHES_DEFAULT
        # structure: tuples of sampling frequency ratios and sets of pitches

    filters = {} # dictionary indexed by sampling frequency ratio. Each item is again a dictionary indexed by pitch, giving a filter coefficient tuple.
    
    for ratio, pitches in ratios_pitches:
        filters[ratio] = {}
        current_fs = float(fs / ratio) # input sampling frequency for the current set of pitches
        nyquist_freq = current_fs / 2
        for pitch in pitches:
            freq = pitch2freq(pitch)
            w = freq / nyquist_freq # omega = normalised frequency
            w_pass = (w * (1 - 1 / (2*Q)), w * (1 + 1 / (2*Q)))
            w_stop = (w * (1 - 1 / Q), w * (1 + 1 / Q))
            n, w_natural = sig.ellipord(w_pass, w_stop, max_loss_pass, min_attenuation_stop)
            coeff_b, coeff_a = sig.ellip(n, max_loss_pass, min_attenuation_stop, w_natural, btype='bandpass') # get filter coefficients
            # note that scipy's ellip differs from matlab's in that it will always generate a lowpass filter by default.
            # btype='bandpass' needs to be passed explicitly!
            filters[ratio][pitch] = (coeff_b, coeff_a)
    
    return filters
    

def audio_to_pitch(audio, filter_bank, window_size=800, window_shift=400):
    # window size: number of samples per window. at 16kHz, a window of 800 samples spans 50ms.
    output = {} # output will be a dictionary indexed by pitch.
    for ratio, filters in filter_bank.items():
        if ratio == 1.0:
            resampled_audio = audio # decimate with ftype='fir' bitches when the ratio is 1. So do nothing in this case.
        else:
            resampled_audio = sig.decimate(audio, ratio, ftype='fir') # scipy's resample is different from matlab's resample...
            
        current_window_size = int(window_size / ratio)
        current_window_shift = int(window_shift / ratio)
            
        # use ftype='fir' instead of the default 'iir' to avoid warnings and bogus values.
        for pitch, filter_coeffs in filters.items():
            coeff_b, coeff_a = filter_coeffs
            out = sig.filtfilt(coeff_b, coeff_a, resampled_audio)
            out_squared = out ** 2 # energy
            
            # summarise over windows:
            windows = np.arange(0, len(out_squared), current_window_shift)
            summarised_output = np.zeros(len(windows))
            for i, s in enumerate(np.arange(0, len(out_squared), current_window_shift)):
                # summarised_output[i] = ratio * np.sum(out_squared[s:s+current_window_size])
                summarised_output[i] = np.mean(out_squared[s:s+current_window_size])
            
            output[pitch] = summarised_output

    return output
    
    
def audio_to_pitch_given_boundaries(audio, filter_bank, window_boundaries):
    """
    like audio_to_pitch, but takes an iterable (window_boundaries) that contains
    2-tuples of the start and end of each window.
    """
    output = {} # output will be a dictionary indexed by pitch.
    for ratio, filters in filter_bank.items():
        if ratio == 1.0:
            resampled_audio = audio # decimate with ftype='fir' bitches when the ratio is 1. So do nothing in this case.
        else:
            resampled_audio = sig.decimate(audio, ratio, ftype='fir') # scipy's resample is different from matlab's resample...
            # use ftype='fir' instead of the default 'iir' to avoid warnings and bogus values.
        
        for pitch, filter_coeffs in filters.items():
            coeff_b, coeff_a = filter_coeffs
            out = sig.filtfilt(coeff_b, coeff_a, resampled_audio)
            out_squared = out ** 2 # energy
            
            # summarise over windows:
            summarised_output = np.zeros(len(window_boundaries))
            for (i, (start, end)) in enumerate(window_boundaries):
                summarised_output[i] = np.mean(out_squared[int(start/ratio):int(end/ratio)])
            
            output[pitch] = summarised_output

    return output
    
    
def pitch_dict_to_array(pitch_dict):
    pitches = sorted(pitch_dict.keys())
    out = np.zeros((len(pitches), len(pitch_dict.values()[0])))
    for i, p in enumerate(pitches):
        out[i] = pitch_dict[p]
    return out

def audio_to_log_pitch_array(audio, filter_bank, window_size=800, window_shift=400, C=10.0**5):
    return loudness_log(audio_to_pitch_array(audio, filter_bank, window_size=window_size, window_shift=window_shift), C=C)
    
def audio_to_pitch_array(audio, filter_bank, window_size=800, window_shift=400):
    pitch_rep = audio_to_pitch(audio, filter_bank, window_size=window_size, window_shift=window_shift)
    return pitch_dict_to_array(pitch_rep)
    
def audio_to_pitch_array_tempo(audio, filter_bank, window_boundaries):
    pitch_rep = audio_to_pitch_given_boundaries(audio, filter_bank, window_boundaries)
    return pitch_dict_to_array(pitch_rep)
    
    
def loudness_log(data, C=10.0**5):
    return np.log(1 + data * C) # convert to a logarithmic loudness scale
 

