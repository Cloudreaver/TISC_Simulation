#!/usr/bin/env python


# File used to generate CW noise at a given frequency
# Then filter and resample if desired

import numpy as np
#from math import sin
from anita_filter import butter_bandpass, butter_bandpass_filter
from scipy.signal import resample
from random import random


def generate_cw(sample_length,upsample=10,sample_frequency=2600000000.0, carrier_frequency=500000000.0,modulation_frequency=1.0, cw_rms=20.0,filter_flag=True):

    upsample_length = sample_length*upsample
    upsample_frequency = sample_frequency*upsample
    
    # Convert peak-to-peak to rms value (0.707 is the RMS of a sine wave with A=1)
    peak_amplitude = cw_rms#/0.707

    # Make large sample length to avoid edge effects
    expanded_sample_length = sample_length * 2
    expanded_upsample_length = upsample_length * 2
    
    # Define the cw array
    upsampled_cw = np.zeros(expanded_upsample_length)
    cw = np.zeros(expanded_sample_length)
    M=1.0 # Envelope Amplitude

    # Fill array with AM wave of given freq and amp
    for i in range(0,int(expanded_upsample_length)):
        upsampled_cw[i] = peak_amplitude*np.sin(2*np.pi*(carrier_frequency/upsample_frequency)*float(i))*(2/np.pi)*(1+M*np.cos(2*np.pi*(modulation_frequency/upsample_frequency)*float(i)))

    # Apply random phase shift to signal
    phase_shift = int((upsample_frequency/modulation_frequency)*random())
    cw = np.roll(cw,phase_shift)
    phase = int(np.random.uniform()*upsample)
    for t in range(0,expanded_sample_length):
        cw[t] = upsampled_cw[t*upsample+phase]

    # Apply bandpass filter is desired
    if filter_flag:
        cw = butter_bandpass_filter(cw)

    # Define start and end of signal window (avoids weird edge effects)
    window_start = expanded_sample_length/3
    window_end = window_start+sample_length

    return cw[window_start:window_end]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sample_length = 80
    upsample = 10
    sample_frequency =    2600000000.0
    carrier_frequency =    500000000.0 # Dominant freq. seen in ANITA3
    modulation_frequency =  1000.0 
    cw_rms = 20.0
    filter_flag = False

    cw = np.zeros(sample_length)
    t = np.zeros(sample_length)

    cw = generate_cw(sample_length,upsample,sample_frequency,carrier_frequency,modulation_frequency,cw_rms,filter_flag)

    plt.figure(1)
    plt.clf()
    #print (1/sample_frequency)*1000000000.0
    
    t = np.linspace(0,(1/sample_frequency)*sample_length*1000000000,sample_length)

    plt.plot(t,cw, label='CW Signal')

    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title(str(carrier_frequency/1000000)+" MHz CW Simulation")

    plt.show()

    
