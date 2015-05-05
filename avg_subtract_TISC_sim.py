#!/usr/bin/env python

# A very simple impulsive signal + noise generator to get the 

import random
#from array import array
import numpy as np
from impulse import impulse_gen, butter_bandpass_filter, butter_bandpass
from noise import generate_noise
from digitizer import digitize
from avg_subtracted_sum_correlator import avg_subtracted_sum_correlate
from cw import generate_cw

def avg_subtract_TISC_sim(SNR,threshold,
             impulse_position,b_input_delay,c_input_delay,num_bits=3,
             noise_sigma=20.0,noise_mean=0.0,
             sample_freq=2600000000.0,TISC_sample_length=16,
             num_samples=74,upsample=1,cw_flag=0,
             peak_amplitude=25.0,carrier_frequency=260000000.0,modulation_frequency=16000000.0,
             seed=5522684,draw_flag=0,digitization_factor=1,
             delay_type_flag=1,
             output_dir="output/",average_signal=150):
   
                           

   
   # Setup
   save_output_flag = 0
   if(save_output_flag):
      outfile = str(output_dir+"/test.root")
   trigger_flag = 0
   #num_bits = 3 # Number of bits available to the digitizer
   num_upsamples = num_samples*upsample
   filter_flag = True
   #sample_frequency = 2800000000.0
   
   # Fill numpy arrays with zeros
   a_input_noise = np.zeros(num_upsamples)
   b_input_noise = np.zeros(num_upsamples)
   c_input_noise = np.zeros(num_upsamples)
   #time = np.zeros(num_samples)
   #upsampled_time = np.zeros(num_upsamples)
   a_input_signal = np.zeros(num_upsamples)
   b_input_signal = np.zeros(num_upsamples)
   c_input_signal = np.zeros(num_upsamples)
   a_input_signal_noise = np.zeros(num_upsamples)
   b_input_signal_noise = np.zeros(num_upsamples)
   c_input_signal_noise = np.zeros(num_upsamples)
   a_dig_waveform = np.zeros(num_samples)
   b_dig_waveform = np.zeros(num_samples)
   c_dig_waveform = np.zeros(num_samples)
   empty_list = np.zeros(num_upsamples)
   new_average_signal =0
 
   #if (updample>1):
      #for i in range(0,num_upsamples):
         #upsampled_time[i] = (i/(sample_freq*upsample))*(10**9)

   #for i in range(0,num_samples):
      #time[i] = (i/(sample_freq))*(10**9)

###################################
   # Generate Thermal Noise
   a_input_noise = generate_noise(num_samples,upsample,noise_mean,noise_sigma,filter_flag)
   b_input_noise = generate_noise(num_samples,upsample,noise_mean,noise_sigma,filter_flag)
   c_input_noise = generate_noise(num_samples,upsample,noise_mean,noise_sigma,filter_flag)
###################################


#####################################
   # Determine RMS of noise and signal amplitude
   noise_rms = np.sqrt(np.mean((a_input_noise-noise_mean)**2,))   
   signal_amp = SNR*2*noise_rms
#####################################
   
   
#################################
   #Generate CW noise if desired
   if cw_flag:
      a_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      a_input_noise += a_input_cw_noise

      b_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      b_input_noise += b_input_cw_noise

      c_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      c_input_noise += c_input_cw_noise
#####################################
 
 
#####################################
   # Generate impulse
   if (SNR != 0):
      # Generate Signal and Amplify
      a_input_signal = impulse_gen(num_samples,impulse_position,upsample,draw_flag=draw_flag,output_dir=output_dir)
      difference=np.amax(a_input_signal)-np.amin(a_input_signal) # Get peak to peak voltage
      a_input_signal *= (1/difference) # Normalize input
      a_input_signal *= signal_amp # Amplify
      #print "SNR: " +str(SNR)
      #print "Noise RMS: " +str(noise_rms)
      #print "Signal Amp: " +str(signal_amp)
      #print "Difference: "+str(difference)
      #print "New signal amp: "+str(np.amax(a_input_signal)-np.amin(a_input_signal))
      b_input_signal = np.concatenate([empty_list[:b_input_delay*upsample],a_input_signal[:num_upsamples-b_input_delay*upsample]])
      c_input_signal = np.concatenate([empty_list[:c_input_delay*upsample],a_input_signal[:num_upsamples-c_input_delay*upsample]])

      # Add the signal to the noise
      a_input_signal_noise = np.add(a_input_noise, a_input_signal)
      b_input_signal_noise = np.add(b_input_noise, b_input_signal)
      c_input_signal_noise = np.add(c_input_noise, c_input_signal)
   else:
      a_input_signal_noise = a_input_noise
      b_input_signal_noise = b_input_noise
      c_input_signal_noise = c_input_noise
##########################################


##########################################
   # Digitized the incoming signal and noise (RITC)
   a_dig_waveform = digitize(a_input_signal_noise,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)
   b_dig_waveform = digitize(b_input_signal_noise,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)
   c_dig_waveform = digitize(c_input_signal_noise,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)

##########################################


##########################################
   # Run the signal through the GLITC module to get trigger
   trigger_flag, max_sum = avg_subtracted_sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,threshold,TISC_sample_length,delay_type_flag=delay_type_flag,average_signal=average_signal)
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

   return trigger_flag, max_sum

if __name__ == '__main__':
   #import ROOT
   import time
   num_samples = 74
   impulse_position = 20
   num_bits = 3
   #signal_amp = 10.0
   draw_flag = True
   b_input_delay = 0
   c_input_delay = 0
   SNR = 5
   threshold = 100
   #num_runs = 100
   upsample = 1
   cw_flag = False
   cw_amplitude = 200.0
   cw_frequency = 260000000.0
   delay_type_flag = 1
   digitization_factor=1.0
   
   print "SNR: "+str(SNR)
   print "Threshold: "+str(threshold)


   # Get impulse signal
   #a_input_signal = np.zeros(num_samples*upsample)
   #a_input_signal = impulse_gen(num_samples,impulse_position,upsample=upsample,draw_flag=draw_flag)
   passedFlag = TISC_sim(SNR,threshold,impulse_position,b_input_delay,c_input_delay,num_bits=num_bits,
                         upsample=upsample,num_samples=num_samples,noise_sigma=100,
                         cw_flag=cw_flag,peak_amplitude=cw_amplitude,carrier_frequency=cw_frequency,modulation_frequency=16000000.0,
                         draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir="output/",delay_type_flag=delay_type_flag)
   print passedFlag
