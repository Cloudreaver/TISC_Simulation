#!/usr/bin/env python

# A very simple impulsive signal + noise generator to get the 

import random
#from array import array
import numpy as np
from impulse import impulse_gen, butter_bandpass_filter, butter_bandpass
from noise import generate_noise
from digitizer import digitize
from sum_correlator import sum_correlate
from cw import generate_cw
from impulse import butter_bandpass_filter
from datetime import datetime

def TISC_sim(SNR,threshold,
             impulse_position,b_input_delay,c_input_delay,num_bits=3,
             noise_sigma=20.0,noise_mean=0.0,
             sample_freq=2600000000.0,TISC_sample_length=16,
             num_samples=74,upsample=10,cw_flag=0,
             peak_amplitude=25.0,carrier_frequency=260000000.0,modulation_frequency=1.0,
             seed=5522684,draw_flag=0,digitization_factor=1,
             delay_type_flag=1,
             output_dir="output/"):
   #start_time = datetime.now()
                           

   
   # Setup
   save_output_flag = 0
   if(save_output_flag):
      outfile = str(output_dir+"/test.root")
   trigger_flag = 0
   #num_bits = 3 # Number of bits available to the digitizer
   filter_flag = False
   #sample_frequency = 2800000000.0
   
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
   empty_list = np.zeros(num_samples)
   #new_average_signal =0
 
   #start_thermal = datetime.now()
###################################
   # Generate Thermal Noise
   a_input_noise = generate_noise(num_samples,noise_mean,noise_sigma,filter_flag)
   b_input_noise = generate_noise(num_samples,noise_mean,noise_sigma,filter_flag)
   c_input_noise = generate_noise(num_samples,noise_mean,noise_sigma,filter_flag)
###################################
   #print "Thermal Noise Generation Took: " +str(datetime.now()-start_thermal)

   #start_rms = datetime.now()
#####################################
   # Determine RMS of noise and signal amplitude
   noise_rms = np.sqrt(np.mean((a_input_noise-noise_mean)**2,))   
   signal_amp = SNR*2*noise_rms
#####################################
   #print "RMS Took: " +str(datetime.now()-start_rms)
   
   #start_cw = datetime.now()
#################################
   #Generate CW & thermal noise
   
   if cw_flag:
      a_input_noise = np.add(a_input_noise,generate_cw(num_samples,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag))

      b_input_noise = np.add(b_input_noise,generate_cw(num_samples,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag))

      c_input_noise = np.add(c_input_noise,generate_cw(num_samples,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag))
   #print "CW Took: " +str(datetime.now()-start_cw)
   #start_filter = datetime.now()
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
      #print "Length of a 1"+ str(len(a_input_signal))
      difference=np.amax(a_input_signal)-np.amin(a_input_signal) # Get peak to peak voltage
      a_input_signal *= (1/difference) # Normalize input
      a_input_signal *= signal_amp # Amplify
      #print "Length of a 2"+ str(len(a_input_signal))
      #print "SNR: " +str(SNR)
      #print "Noise RMS: " +str(noise_rms)
      #print "Signal Amp: " +str(signal_amp)
      #print "Difference: "+str(difference)
      #print "New signal amp: "+str(np.amax(a_input_signal)-np.amin(a_input_signal))
      #print len(empty_list[:b_input_delay])+len(a_input_signal[:num_samples-b_input_delay])
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
   #print "Signal Took: " +str(datetime.now()-start_signal)
   #start_dig = datetime.now()

##########################################
   # Digitized the incoming signal and noise (RITC)
   a_dig_waveform = digitize(a_input_signal,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)
   b_dig_waveform = digitize(b_input_signal,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)
   c_dig_waveform = digitize(c_input_signal,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)

##########################################
   #print "Dig Took: " +str(datetime.now()-start_dig)
   #start_corr = datetime.now()

##########################################
   # Run the signal through the GLITC module to get trigger
   trigger_flag, max_sum = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,threshold,TISC_sample_length,delay_type_flag=delay_type_flag)
#########################################
   #print "Corr Took: " +str(datetime.now()-start_corr)

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
   digitization_factor=1.0
   
   #print "SNR: "+str(SNR)
   #print "Threshold: "+str(threshold)


   # Get impulse signal
   #a_input_signal = np.zeros(num_samples*upsample)
   #a_input_signal = impulse_gen(num_samples,impulse_position,upsample=upsample,draw_flag=draw_flag)
   passedFlag, max_sum = TISC_sim(SNR,threshold,impulse_position,b_input_delay,c_input_delay,num_bits=num_bits,
                         upsample=upsample,num_samples=num_samples,noise_sigma=noise_sigma,
                         cw_flag=cw_flag,peak_amplitude=cw_amplitude,carrier_frequency=cw_frequency,modulation_frequency=modulation_frequency,
                         draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir="output/",delay_type_flag=delay_type_flag)
   print passedFlag
