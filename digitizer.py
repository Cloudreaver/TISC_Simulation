#!/usr/bin/env python

# Simulates the RITC output 

import random
import numpy as np

def digitize(input_array,num_samples,num_bits,digitization_factor=32.0,debug=False):
   
   #noise_rms *= digitization_factor
         
   # Fill numpy arrays with zeros
   dig_waveform = np.zeros(num_samples)
   
   # Convert to 3 bit values
   for index in range(0,num_samples):
      for bit_index in range(0,2**(num_bits-1)):
         if input_array[index] > (bit_index)*(digitization_factor): 
            dig_waveform[index] = bit_index+0.5
         elif input_array[index] < ((-bit_index)*(digitization_factor)): 
            dig_waveform[index] = -bit_index-0.5
      
         
   return dig_waveform

if __name__ == '__main__':
   import numpy as np
   from noise import generate_noise
   import matplotlib.pyplot as plt

   debug=True
   num_samples = 128
   num_bits = 3
   noise_sigma = 32.0
   digitization_factor = 32.0
   sample_frequency = 2600000000.0

   sample_noise = np.zeros(num_samples)
   dig_waveform = np.zeros(num_samples)
   sine_waveform = np.zeros(num_samples)

   sample_noise = generate_noise(num_samples,noise_sigma=noise_sigma,filter_flag=1)
   
   for i in range(0,int(num_samples)):
      sine_waveform[i] = 3.0*noise_sigma*np.sin(2*np.pi*50000000.0*(float(i)/sample_frequency))
         
   sample = sine_waveform + sample_noise
   #sample = sample_noise
   dig_waveform = digitize(sample,num_samples,num_bits,digitization_factor,debug=debug)
   
   t = np.linspace(0,(1/2800000000.0)*num_samples*(10**9),num_samples)

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
