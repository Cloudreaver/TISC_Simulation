#!/usr/bin/env python

# A very simple impulsive signal + noise generator to get the 

import random
#from array import array
import numpy as np
from impulse import impulse_gen
from anita_filter import butter_bandpass, butter_bandpass_filter
from noise import generate_noise
from digitizer import digitize
from sum_correlator import sum_correlate
from cw import generate_cw
from datetime import datetime

def TISC_sim(SNR,threshold,
             impulse_position,b_input_delay,c_input_delay,num_bits=3,
             noise_sigma=20.0,noise_mean=0.0,
             sample_freq=2600000000.0,TISC_sample_length=16,
             num_samples=74,upsample=10,cw_flag=0,
             peak_amplitude=25.0,carrier_frequency=260000000.0,modulation_frequency=1.0,
             seed=5522684,draw_flag=0,digitization_factor=20.0,
             delay_type_flag=1,
             output_dir="output/",average_subtract_flag=0,correlation_mean=np.zeros(44),trial_run_number=1):                         

   #print correlation_mean
   # Setup
   save_output_flag = 0
   if(save_output_flag):
      outfile = str(output_dir+"/test.root")
   trigger_flag = 0
   #num_bits = 3 # Number of bits available to the digitizer
   filter_flag = False

   # Fill numpy arrays with zeros
   a_input_noise = np.zeros(num_samples)
   b_input_noise = np.zeros(num_samples)
   c_input_noise = np.zeros(num_samples)
   a_input_signal = np.zeros(num_samples)
   b_input_signal = np.zeros(num_samples)
   c_input_signal = np.zeros(num_samples)
   a_dig_waveform = np.zeros(num_samples)
   b_dig_waveform = np.zeros(num_samples)
   c_dig_waveform = np.zeros(num_samples)
   cw_noise = np.zeros(num_samples)
   empty_list = np.zeros(num_samples)

###################################
   # Generate Thermal Noise
   a_input_noise = generate_noise(num_samples,noise_mean,noise_sigma,filter_flag)
   b_input_noise = generate_noise(num_samples,noise_mean,noise_sigma,filter_flag)
   c_input_noise = generate_noise(num_samples,noise_mean,noise_sigma,filter_flag)
###################################

#####################################
   # Determine signal amplitude    
   signal_amp = SNR*2*noise_sigma
#####################################

#################################
   #Generate CW & thermal noise
   
   if cw_flag:
      cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      a_input_noise = np.add(a_input_noise,cw_noise)
      #cw_noise = generate_cw(num_samples,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      b_input_noise = np.add(b_input_noise,cw_noise)
      #cw_noise = generate_cw(num_samples,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      c_input_noise = np.add(c_input_noise,cw_noise)

#####################################

   # Filter the noise
   a_input_noise = butter_bandpass_filter(a_input_noise)
   b_input_noise = butter_bandpass_filter(b_input_noise)
   c_input_noise = butter_bandpass_filter(c_input_noise)
   #print "Filter Took: " +str(datetime.now()-start_filter)
   #start_signal = datetime.now()
#####################################
   # Generate impulse
   if (SNR != 0):
      # Generate Signal and Amplify
      a_input_signal = impulse_gen(num_samples,impulse_position,upsample,draw_flag=draw_flag,output_dir=output_dir)
      difference=np.amax(a_input_signal)-np.amin(a_input_signal) # Get peak to peak voltage
      a_input_signal *= (1/difference) # Normalize input
      a_input_signal *= signal_amp # Amplify
      b_input_signal = np.concatenate([empty_list[:b_input_delay],a_input_signal[:num_samples-b_input_delay]])
      c_input_signal = np.concatenate([empty_list[:c_input_delay],a_input_signal[:num_samples-c_input_delay]])

      # Add the signal to the noise
      a_input_signal = np.add(a_input_noise, a_input_signal)
      b_input_signal = np.add(b_input_noise, b_input_signal)
      c_input_signal = np.add(c_input_noise, c_input_signal)
   else:
      a_input_signal = a_input_noise
      b_input_signal = b_input_noise
      c_input_signal = c_input_noise
##########################################


##########################################
   # Digitized the incoming signal and noise (RITC)
   a_dig_waveform = digitize(a_input_signal,num_samples,num_bits,digitization_factor)
   b_dig_waveform = digitize(b_input_signal,num_samples,num_bits,digitization_factor)
   c_dig_waveform = digitize(c_input_signal,num_samples,num_bits,digitization_factor)

##########################################


##########################################
   # Run the signal through the GLITC module to get trigger
   if (average_subtract_flag):
      trigger_flag, max_sum, as_max_sum, correlation_mean = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,
                                                threshold,TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag,correlation_mean=correlation_mean,trial_run_number=trial_run_number)
   else:
      trigger_flag, max_sum = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,
                                 threshold,TISC_sample_length,delay_type_flag=delay_type_flag,
                                 average_subtract_flag=average_subtract_flag,correlation_mean=correlation_mean)
#########################################


#########################################
   # Output data
   if(save_output_flag):
      # Now to more ROOT stuff
      rf_tree.Fill()
      f.Write()
      f.Close()

   if draw_flag:
      dummy = raw_input('Press any key to close')
      
   #print "Everything took: " +str(datetime.now()-start_time)
   if (average_subtract_flag):
      return trigger_flag, max_sum, as_max_sum, correlation_mean
   else:
      return trigger_flag, max_sum

if __name__ == '__main__':
   #import ROOT
   import time
   num_samples = 74
   impulse_position = 20
   num_bits = 3
   #signal_amp = 10.0
   draw_flag = False
   b_input_delay = 0
   c_input_delay = 0
   SNR = 5
   threshold = 100
   noise_sigma = 20
   #num_runs = 100
   upsample = 1
   cw_flag = True
   cw_amplitude = 1.0*noise_sigma
   cw_frequency = 260000000.0
   modulation_frequency = 1.0
   delay_type_flag = 1
   digitization_factor=20.0
   average_subtract_flag = 1
   #global correlation_mean
   correlation_mean = np.zeros(44)
   correlation_mean.fill(100)
   
   #print correlation_mean
   
    
   
   #print "SNR: "+str(SNR)
   #print "Threshold: "+str(threshold)


   # Get impulse signal
   #a_input_signal = np.zeros(num_samples*upsample)
   #a_input_signal = impulse_gen(num_samples,impulse_position,upsample=upsample,draw_flag=draw_flag)
   for i in range(0,10):
      passedFlag, max_sum, as_max_sum, correlation_mean = TISC_sim(SNR,threshold,impulse_position,b_input_delay,c_input_delay,num_bits=num_bits,
                         upsample=upsample,num_samples=num_samples,noise_sigma=noise_sigma,
                         cw_flag=cw_flag,peak_amplitude=cw_amplitude,carrier_frequency=cw_frequency,modulation_frequency=modulation_frequency,
                         draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir="output/",delay_type_flag=delay_type_flag,
                         average_subtract_flag=average_subtract_flag,correlation_mean=correlation_mean,trial_run_number=i+1)
   print passedFlag
