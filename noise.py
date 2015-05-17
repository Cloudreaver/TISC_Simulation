#!/usr/bin/env python

import numpy as np
import random
from anita_filter import butter_bandpass, butter_bandpass_filter
from scipy.signal import resample

def generate_noise(sample_length,noise_mean=0.0, noise_sigma=20.0,filter_flag=1):
    random.seed()

    # Define the noise array
    noise = np.zeros(sample_length)
    
    # Fill array with random gaussian noise
    # May need to switch to using Rayleigh noise
    for i in range(0, sample_length):
        noise[i] = random.gauss(0,noise_sigma)

    if filter_flag:
        noise = butter_bandpass_filter(noise)+noise_mean

    return noise

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	import numpy as np
	sample_length = 74
   
	#upsample = 1
	noise_sigma = 100
	noise_mean = 0
	filter_flag = True
        run_number = 1

	#upsample_length = sample_length*upsample

	noise = np.zeros(sample_length)
        for i in range(0,run_number):
            noise += generate_noise(sample_length,noise_mean,noise_sigma,filter_flag)
    

        noise /= run_number
	t = np.linspace(0,(1/2800000000.0)*sample_length*(10**9),sample_length)
    
	plt.figure(1)
	plt.clf()
        plt.axis([0.0,30.0,-500.0,500.0])
        ax= plt.gca()
        ax.set_autoscale_on(False)
	plt.plot(t,noise,label="Filtered,Resampled Noise")
	plt.title("Simulated Gaussian Noise")
	plt.xlabel("Time [ns]")
	plt.ylabel("Voltage [mV]")
	plt.text(100,450, "Mean = "+str(noise_mean)+"mV, Sigma = "+str(noise_sigma)+"mV")
	plt.show()

    
