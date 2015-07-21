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
             b_input_delay,c_input_delay,num_bits=3,
             noise_sigma=20.0,
             sample_freq=2600000000.0,TISC_sample_length=16,
             num_samples=74,upsample=10,cw_flag=0,
             cw_rms=25.0,carrier_frequency=260000000.0,modulation_frequency=1.0,
             seed=5522684,draw_flag=0,digitization_factor=20.0,
             delay_type_flag=1,
             output_dir="output/",average_subtract_flag=0,correlation_mean=np.zeros(44),trial_run_number=1):                         

   #print b_input_delay
   # Setup
   save_output_flag = 0
   #if(save_output_flag):
      #outfile = str(output_dir+"/test.root")
   trigger_flag = 0
   #num_bits = 3 # Number of bits available to the digitizer
   filter_flag = False
   #print SNR

   # Fill numpy arrays with zeros
   a_input_noise = np.zeros(num_samples)
   b_input_noise = np.zeros(num_samples)
   c_input_noise = np.zeros(num_samples)
   a_input_noise_test = np.zeros(num_samples)
   b_input_noise_test = np.zeros(num_samples)
   c_input_noise_test = np.zeros(num_samples)
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
   a_input_noise = generate_noise(num_samples,noise_sigma,filter_flag)
   b_input_noise = generate_noise(num_samples,noise_sigma,filter_flag)
   c_input_noise = generate_noise(num_samples,noise_sigma,filter_flag)
   #a_input_noise_test = a_input_noise
   #b_input_noise_test = b_input_noise
   #c_input_noise_test = c_input_noise
###################################

   #print a_input_noise[0]
   #print b_input_noise[0]
   #print c_input_noise[0]
#####################################
   # Determine signal amplitude    
   signal_amp = SNR*2*noise_sigma
#####################################

#################################
   #Generate CW & thermal noise
   
   if cw_flag:
      cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,cw_rms,filter_flag)
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
      a_input_signal = impulse_gen(num_samples,upsample,draw_flag=draw_flag,output_dir=output_dir)
      difference=np.amax(a_input_signal)-np.amin(a_input_signal) # Get peak to peak voltage
      a_input_signal *= (1/difference) # Normalize input
      a_input_signal *= signal_amp # Amplify
      b_input_signal = np.concatenate([a_input_signal[:num_samples+b_input_delay],empty_list[:(-1)*b_input_delay]])
      c_input_signal = np.concatenate([a_input_signal[:num_samples+c_input_delay],empty_list[:(-1)*c_input_delay]])

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
   #print a_dig_waveform

##########################################
   # Run the signal through the GLITC module to get trigger
   if (average_subtract_flag):
      trigger_flag, max_sum, as_max_sum, correlation_mean, test_sum, as_test_sum = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,
                                                threshold,TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag,correlation_mean=correlation_mean,trial_run_number=trial_run_number)
   else:
      trigger_flag, max_sum, test_sum = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,
                                 threshold,TISC_sample_length,delay_type_flag=delay_type_flag,
                                 average_subtract_flag=average_subtract_flag,correlation_mean=correlation_mean)
   """
   if (max_sum>800):
      import matplotlib.pyplot as plt
      #print a_dig_waveform
      #print b_dig_waveform
      #print c_dig_waveform
      #print np.add(np.add(a_dig_waveform,b_dig_waveform),c_dig_waveform)
      #print (np.add(np.add(a_dig_waveform,b_dig_waveform),c_dig_waveform))**2
      #print np.sum((np.add(np.add(a_dig_waveform,b_dig_waveform),c_dig_waveform))**2)
      time = np.linspace(0.0,((num_samples*(10**9))/sample_freq), num_samples)
      plt.figure(1)
      plt.clf()
      plt.plot(time,a_dig_waveform)
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("Ch A")
      plt.savefig(output_dir+"/ch_A_large_correlation.png")
      plt.figure(2)
      plt.clf()
      plt.plot(time,b_dig_waveform)
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("Ch B")
      plt.savefig(output_dir+"/ch_B_large_correlation.png")
      plt.figure(3)
      plt.clf()
      plt.plot(time,c_dig_waveform)
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("Ch C")
      plt.savefig(output_dir+"/ch_C_large_correlation.png")
      plt.figure(4)
      plt.clf()
      plt.plot(time,np.add(np.add(a_dig_waveform,b_dig_waveform),c_dig_waveform))
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("ABC Added")
      plt.savefig(output_dir+"/ABC_large_correlation.png")
      plt.figure(5)
      plt.clf()
      plt.plot(time,(np.add(np.add(a_dig_waveform,b_dig_waveform),c_dig_waveform))**2)
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("ABC Square Added")
      plt.savefig(output_dir+"/ABC_square_large_correlation.png")
      plt.figure(6)
      plt.clf()
      plt.plot(time,a_input_noise_test)
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("Ch A Thermal Noise")
      plt.savefig(output_dir+"/ch_A_thermal_large_correlation.png")
      plt.figure(7)
      plt.clf()
      plt.plot(time,b_input_noise_test)
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("Ch B Thermal Noise")
      plt.savefig(output_dir+"/ch_B_thermal_large_correlation.png")
      plt.figure(8)
      plt.clf()
      plt.plot(time,c_input_noise_test)
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("Ch C Thermal Noise")
      plt.savefig(output_dir+"/ch_C_thermal_large_correlation.png")
      plt.figure(9)
      plt.clf()
      plt.plot(time,cw_noise)
      plt.xlabel("Time [ns]")
      plt.ylabel("Amplitude [unitless]")
      plt.title("CW Noise")
      plt.savefig(output_dir+"/cw_large_correlation.png")
      plt.show()
      """
#########################################


#########################################
   # Output data
   #if(save_output_flag):
      # Now to more ROOT stuff
      #rf_tree.Fill()
      #f.Write()
      #f.Close()

   if draw_flag:
      dummy = raw_input('Press any key to close')
      
   #print "Everything took: " +str(datetime.now()-start_time)
   if (average_subtract_flag):
      return trigger_flag, max_sum, as_max_sum, correlation_mean, test_sum, as_test_sum
   else:
      return trigger_flag, max_sum

if __name__ == '__main__':
   #import ROOT
   import time
   num_samples = 80
   impulse_position = 0
   num_bits = 3
   #signal_amp = 10.0
   draw_flag = False
   # These delays should be negative, since A is the top antenna
   # and we expect an upgoing signal
   b_input_delay = -8
   c_input_delay = -7
   SNR = 5
   threshold = 100
   noise_sigma = 20
   #num_runs = 100
   upsample = 1
   cw_flag = True
   cw_rms = 1.0*noise_sigma
   cw_frequency = 260000000.0
   modulation_frequency = 1.0
   delay_type_flag = 1
   digitization_factor=20.0
   average_subtract_flag = 1
   #global correlation_mean
   correlation_mean = np.zeros(44)
   correlation_mean.fill(100)
   
   #print correlation_mean
   
    
   
   print "SNR: "+str(SNR)
   print "Threshold: "+str(threshold)


   # Get impulse signal
   #a_input_signal = np.zeros(num_samples*upsample)
   #a_input_signal = impulse_gen(num_samples,impulse_position,upsample=upsample,draw_flag=draw_flag)
   for i in range(0,10):
      passedFlag, max_sum, as_max_sum, correlation_mean, d1, d2 = TISC_sim(SNR,threshold,b_input_delay,c_input_delay,num_bits=num_bits,
                         upsample=upsample,num_samples=num_samples,noise_sigma=noise_sigma,
                         cw_flag=cw_flag,cw_rms=cw_rms,carrier_frequency=cw_frequency,modulation_frequency=modulation_frequency,
                         draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir="output/",delay_type_flag=delay_type_flag,
                         average_subtract_flag=average_subtract_flag,correlation_mean=correlation_mean,trial_run_number=(i+1))
   print max_sum
   print as_max_sum
