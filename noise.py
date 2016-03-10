#!/usr/bin/env python

import numpy as np
import random
from anita_filter import butter_bandpass, butter_bandpass_filter
from scipy.signal import resample

def generate_noise(sample_length, noise_sigma=32.0,filter_flag=0,seed=None):
    random.seed(seed)

    # Define the noise array
    noise = np.zeros(sample_length)
    
    # Fill array with random gaussian noise
    for i in range(0, sample_length):
        noise[i] = np.random.normal(0,noise_sigma)

    if filter_flag:
        noise = butter_bandpass_filter(noise)

    return noise

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import numpy as np
	sample_length = 2560
   
	noise_sigma = 32.0
	filter_flag = True

	noise = np.zeros(sample_length)
	noise = generate_noise(sample_length,noise_sigma,filter_flag,seed=1)
	t = np.linspace(0,(1/2800000000.0)*sample_length*(10**9),sample_length)
    
	plt.figure(1)
	plt.clf()
        #plt.axis([0.0,30.0,-100.0,100.0])
        #ax= plt.gca()
        #ax.set_autoscale_on(False)
	plt.plot(t,noise,label="Filtered,Resampled Noise")
        plt.axhline(noise_sigma,label="Thermal Noise RMS",color='r')
	plt.axhline((-1)*noise_sigma,color='r')
        plt.title("Simulated Gaussian Noise")
	plt.xlabel("Time [ns]")
	plt.ylabel("Voltage [mV]")
	plt.text(100,450, "Sigma = "+str(noise_sigma)+"mV")
	plt.show()

    
