#!/usr/bin/env python

# Simulates the RITC output 

import random
import numpy as np

def digitize(input_array,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor=1):
   
   noise_rms *= digitization_factor
         
   # Fill numpy arrays with zeros
   dig_waveform = np.zeros(num_samples)
   upsample = 1
   # Convert to 3 bit values
   for index in range(0,num_samples):
      for bit_index in range(0,2**(num_bits-1)):
         if (upsample > 1):
            downsample_random = random.randrange(0,upsample-1)
         else:
            downsample_random = 0
         if input_array[(index*upsample)+downsample_random] > (bit_index)*(noise_rms)+noise_mean: 
            dig_waveform[index] = bit_index+0.5
         elif input_array[(index*upsample)+downsample_random] < ((-bit_index)*(noise_rms)+noise_mean): 
            dig_waveform[index] = -bit_index-0.5
      
         
   return dig_waveform

if __name__ == '__main__':
   import numpy as np
   from noise import generate_noise
   import matplotlib.pyplot as plt

   num_samples = 256
   upsample = 1
   num_bits = 3
   noise_mean = 450
   noise_sigma = 100
   digitization_factor = 1
   sample_frequency = 2600000000.0

   num_upsamples = num_samples*upsample
   sample_noise = np.zeros(num_upsamples)
   dig_waveform = np.zeros(num_samples)
   sine_waveform = np.zeros(num_upsamples)

   sample_noise = generate_noise(num_upsamples,upsample,noise_mean,noise_sigma,1)
   noise_rms = np.sqrt(np.mean((sample_noise-noise_mean)**2))
   #sample_noise[4] = noise_mean +(noise_rms*10)
   
   for i in range(0,int(num_upsamples)):
      sine_waveform[i] = 3.0*noise_sigma*np.sin(2*np.pi*50000000.0*(float(i)/sample_frequency))
         
   sample = sine_waveform + sample_noise
   dig_waveform = digitize(sample,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)
   
   t = np.linspace(0,(1/2800000000.0)*num_samples*(10**9),upsample*num_samples)

   plt.figure()
   plt.clf()
   plt.plot(t,dig_waveform,label="Digitized Noise")
   plt.title("Digitized Noise")
   plt.xlabel("Time [ns]")
   plt.ylabel("Voltage [mV]")
   plt.show()



__author__ = "Ryan Hupe"
__copyright__ = "Copyright 2014, The ANITA Project"
__credits__ = ["Ryan Hupe","Jim Beatty","Amy Connolly","Patrick Allison"]

__license__ = "GPL"
__version__ = "1.2"
__maintainer__ = "Ryan Hupe"
__email__ = "ryanhupe@gmail.com"
__status__ = "Development"
