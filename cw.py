#!/usr/bin/env python


# File used to generate CW noise at a given frequency
# Then filter and resample if desired

import numpy as np
#from math import sin
from impulse import butter_bandpass_filter
from scipy.signal import resample
from random import random


def generate_cw(sample_length,sample_frequency, carrier_frequency,modulation_frequency, peak_amplitude,filter_flag=True):



    # Make large sample length to avoid edge effects
    expanded_sample_length = sample_length * 2

    #upsample_length = sample_length*upsample

    #expanded_upsample_length = upsample_length * 2
    
    # Define the cw array
    cw = np.zeros(expanded_sample_length)
    #resampled_cw = np.zeros(expanded_sample_length)
    M=0.5


    
    # Fill array with sine wave of given freq and amp
    for i in range(0,int(expanded_sample_length)):
        cw[i] = peak_amplitude*np.sin(2*np.pi*(carrier_frequency/sample_frequency)*float(i))*(2/np.pi)*(1+M*np.cos(2*np.pi*(modulation_frequency/sample_frequency)*float(i)))

    # Apply bandpass filter is desired
    if filter_flag:
        cw = butter_bandpass_filter(cw)

    # Apply random phase shift to signal
    phase_shift = int((sample_frequency/modulation_frequency)*random())
    cw = np.roll(cw,phase_shift)

    # Define start and end of signal window
    window_start = expanded_sample_length/3
    window_end = window_start+sample_length

    # Upsample cw if desired
    #if(upsample > 1.0):
        #resampled_cw = resample(cw,expanded_upsample_length)
        #return resampled_cw[window_start:window_end]
    #else:
    return cw[window_start:window_end]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sample_length = 74
    upsample = 1
    sample_frequency =    2800000000.0
    carrier_frequency =    260000000.0 # Dominant freq. seen in ANITA3
    modulation_frequency =  1.0 
    peak_amplitude = 25.0
    filter_flag = True

    upsample_length = sample_length*upsample

    cw = np.zeros(sample_length)
    t = np.zeros(sample_length)

    cw = generate_cw(sample_length,sample_frequency,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)

    plt.figure(1)
    plt.clf()
    #print (1/sample_frequency)*1000000000.0
    
    t = np.linspace(0,(1/sample_frequency)*sample_length*1000000000,sample_length)

    plt.plot(t,cw, label='CW Signal')

    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title(str(carrier_frequency/1000000)+" MHz CW Simulation")

    plt.show()

    
