#!/usr/bin/env python

import random
from array import array
import numpy as np
from impulse import impulse_gen
from anita_filter import butter_bandpass, butter_bandpass_filter
from noise import generate_noise
from digitizer import digitize
from sum_correlator import sum_correlate
from cw import generate_cw
from simulation_parameters import sim_param

def TISC_sim(sim_param,SNR,seed=10):
   if(sim_param.debug): import matplotlib.pyplot as plt
   
   
   # Setup
   trigger_flag = 0
   filter_flag = False
   def_max_sum = 0.0
   def_as_max_sum = 0.0
   ghi_max_sum = 0.0
   ghi_as_max_sum = 0.0
   timestep = 1.0/sim_param.sample_frequency
   
   
   # Phi sectors have alternating baselines
   if(sim_param.boresight==0):
      abc_impulse_amp = 1.000
      def_impulse_amp = 0.835
      ghi_impulse_amp = 0.776
      if(sim_param.baseline==0):
         abc_baseline = 0
         def_baseline = 1
         ghi_baseline = 1
      elif(sim_param.baseline==1):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 0
   elif(sim_param.boresight==1):
      abc_impulse_amp = 0.962
      def_impulse_amp = 0.885
      ghi_impulse_amp = 0.650
      if(sim_param.baseline==0):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 1
      elif(sim_param.baseline==1):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 0
   
   # Fill numpy arrays with zeros
   a_input_noise = np.zeros(sim_param.num_samples)
   b_input_noise = np.zeros(sim_param.num_samples)
   c_input_noise = np.zeros(sim_param.num_samples)
   d_input_noise = np.zeros(sim_param.num_samples)
   e_input_noise = np.zeros(sim_param.num_samples)
   f_input_noise = np.zeros(sim_param.num_samples)
   g_input_noise = np.zeros(sim_param.num_samples)
   h_input_noise = np.zeros(sim_param.num_samples)
   i_input_noise = np.zeros(sim_param.num_samples)
   time = np.zeros(sim_param.num_samples)
   sim_param.upsampled_time = np.zeros(sim_param.num_samples)
   a_input_signal = np.zeros(sim_param.num_samples)
   b_input_signal = np.zeros(sim_param.num_samples)
   c_input_signal = np.zeros(sim_param.num_samples)
   d_input_signal = np.zeros(sim_param.num_samples)
   e_input_signal = np.zeros(sim_param.num_samples)
   f_input_signal = np.zeros(sim_param.num_samples)
   g_input_signal = np.zeros(sim_param.num_samples)
   h_input_signal = np.zeros(sim_param.num_samples)
   i_input_signal = np.zeros(sim_param.num_samples)
   a_input_signal_noise = np.zeros(sim_param.num_samples)
   b_input_signal_noise = np.zeros(sim_param.num_samples)
   c_input_signal_noise = np.zeros(sim_param.num_samples)
   d_input_signal_noise = np.zeros(sim_param.num_samples)
   e_input_signal_noise = np.zeros(sim_param.num_samples)
   f_input_signal_noise = np.zeros(sim_param.num_samples)
   g_input_signal_noise = np.zeros(sim_param.num_samples)
   h_input_signal_noise = np.zeros(sim_param.num_samples)
   i_input_signal_noise = np.zeros(sim_param.num_samples)
   a_dig_waveform = np.zeros(sim_param.num_samples)
   b_dig_waveform = np.zeros(sim_param.num_samples)
   c_dig_waveform = np.zeros(sim_param.num_samples)
   d_dig_waveform = np.zeros(sim_param.num_samples)
   e_dig_waveform = np.zeros(sim_param.num_samples)
   f_dig_waveform = np.zeros(sim_param.num_samples)
   g_dig_waveform = np.zeros(sim_param.num_samples)
   h_dig_waveform = np.zeros(sim_param.num_samples)
   i_dig_waveform = np.zeros(sim_param.num_samples)
   empty_list = np.zeros(sim_param.num_samples)


###################################
   # Generate Thermal Noise
   a_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed)
   b_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed+1)
   c_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed+2)
   d_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed+3)
   e_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed+4)
   f_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed+5)
   g_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed+6)
   h_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed+7)
   i_input_noise = generate_noise(sim_param.num_samples,sim_param.noise_sigma,seed=seed+8)
###################################

   
#################################
   #Generate CW noise if desired
   if sim_param.cw_flag:
      cw_noise = generate_cw(sim_param.num_samples,sim_param.upsample,sim_param.sample_frequency,sim_param.carrier_frequency,sim_param.modulation_frequency,sim_param.cw_amplitude,sim_param.filter_flag)
      a_input_noise = np.add(a_input_noise,cw_noise)
      b_input_noise = np.add(cw_noise,b_input_noise)
      c_input_noise = np.add(cw_noise,c_input_noise)
      d_input_noise = np.add(cw_noise,d_input_noise)
      e_input_noise = np.add(cw_noise,e_input_noise)
      f_input_noise = np.add(cw_noise,f_input_noise)
      g_input_noise = np.add(cw_noise,g_input_noise)
      h_input_noise = np.add(cw_noise,h_input_noise)
      i_input_noise = np.add(cw_noise,i_input_noise)

#####################################
   
   # Filter noise
   a_input_noise = butter_bandpass_filter(a_input_noise)
   b_input_noise = butter_bandpass_filter(b_input_noise)
   c_input_noise = butter_bandpass_filter(c_input_noise)
   d_input_noise = butter_bandpass_filter(d_input_noise)
   e_input_noise = butter_bandpass_filter(e_input_noise)
   f_input_noise = butter_bandpass_filter(f_input_noise)
   g_input_noise = butter_bandpass_filter(g_input_noise)
   h_input_noise = butter_bandpass_filter(h_input_noise)
   i_input_noise = butter_bandpass_filter(i_input_noise)
   
   if(sim_param.debug):
      time = np.linspace(0.0,timestep*sim_param.num_samples,sim_param.num_samples)
      plt.plot(time,a_input_noise,time,b_input_noise,time,c_input_noise)
      plt.title("Noise")
      plt.show()
 
#####################################
   # Generate impulse
   if (SNR != 0):
      # Determine RMS of noise and signal amplitude   
      signal_amp = SNR*2*sim_param.noise_sigma
      
      # Generate Signal and Amplify
      a_input_signal = impulse_gen(sim_param)
      difference=np.amax(a_input_signal)-np.amin(a_input_signal) # Get peak to peak voltage
      a_input_signal *= (1/difference) # Normalize input
      a_input_signal *= signal_amp # Amplify
      b_input_signal = np.concatenate([a_input_signal[-sim_param.b_input_delay:],empty_list[:(-1)*sim_param.b_input_delay]])
      c_input_signal = np.concatenate([a_input_signal[-sim_param.c_input_delay:],empty_list[:(-1)*sim_param.c_input_delay]])
      
      
      # Account for Seavey angular response
      a_input_signal =a_input_signal*abc_impulse_amp
      b_input_signal =b_input_signal*abc_impulse_amp
      c_input_signal =c_input_signal*abc_impulse_amp
      
      d_input_signal =a_input_signal*def_impulse_amp
      e_input_signal =b_input_signal*def_impulse_amp
      f_input_signal =c_input_signal*def_impulse_amp
      
      g_input_signal =a_input_signal*ghi_impulse_amp
      h_input_signal =b_input_signal*ghi_impulse_amp
      i_input_signal =c_input_signal*ghi_impulse_amp
      

      if(sim_param.debug):
         time = np.linspace(0.0,timestep*sim_param.num_samples,sim_param.num_samples)
         plt.plot(time,a_input_signal,time,b_input_signal,time,c_input_signal)
         plt.plot(time,d_input_signal,time,e_input_signal,time,f_input_signal)
         plt.plot(time,g_input_signal,time,h_input_signal,time,i_input_signal)
         plt.title("impulse")
         plt.show()  
      
      # Add the signal to the noise
      a_input_signal_noise = np.add(a_input_noise, a_input_signal)
      b_input_signal_noise = np.add(b_input_noise, b_input_signal)
      c_input_signal_noise = np.add(c_input_noise, c_input_signal)
      d_input_signal_noise = np.add(d_input_noise, d_input_signal)
      e_input_signal_noise = np.add(e_input_noise, e_input_signal)
      f_input_signal_noise = np.add(f_input_noise, f_input_signal)
      g_input_signal_noise = np.add(g_input_noise, g_input_signal)
      h_input_signal_noise = np.add(h_input_noise, h_input_signal)
      i_input_signal_noise = np.add(i_input_noise, i_input_signal)
      
   else:
      # No impulse, just copy the noise over
      a_input_signal_noise = a_input_noise
      b_input_signal_noise = b_input_noise
      c_input_signal_noise = c_input_noise
      d_input_signal_noise = d_input_noise
      e_input_signal_noise = e_input_noise
      f_input_signal_noise = f_input_noise
      g_input_signal_noise = g_input_noise
      h_input_signal_noise = h_input_noise
      i_input_signal_noise = i_input_noise
      
      if(sim_param.debug):
         time = np.linspace(0.0,timestep*sim_param.num_samples,sim_param.num_samples)
         #plt.plot(time,a_input_signal_noise,time,b_input_signal_noise,time,c_input_signal_noise)
         #plt.plot(time,d_input_signal_noise,time,e_input_signal_noise,time,f_input_signal_noise)
         plt.plot(time,g_input_signal_noise,time,h_input_signal_noise,time,i_input_signal_noise)
         plt.title("impulse+noise")
         plt.show()
##########################################
   
##########################################
   # Digitized the incoming signal and noise (RITC)
   a_dig_waveform = digitize(a_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
   b_dig_waveform = digitize(b_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
   c_dig_waveform = digitize(c_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
   d_dig_waveform = digitize(d_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
   e_dig_waveform = digitize(e_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
   f_dig_waveform = digitize(f_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
   g_dig_waveform = digitize(g_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
   h_dig_waveform = digitize(h_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
   i_dig_waveform = digitize(i_input_signal_noise,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)

##########################################
   
   if(sim_param.debug):
      time = np.linspace(0.0,timestep*sim_param.num_samples,sim_param.num_samples)
      #plt.plot(time,a_dig_waveform,time,b_dig_waveform,time,c_dig_waveform)
      #plt.plot(time,d_dig_waveform,time,e_dig_waveform,time,f_dig_waveform)
      plt.plot(time,g_dig_waveform,time,h_dig_waveform,time,i_dig_waveform)
      plt.title("Digitized")
      plt.show()
   
   
##########################################
   # Run the signal through the GLITC module to get trigger

   abc_trigger_flag, abc_max_sum,abc_angle,d1 = sum_correlate(sim_param,a_dig_waveform,b_dig_waveform,c_dig_waveform,abc_baseline)
   def_trigger_flag, def_max_sum,def_angle,d1 = sum_correlate(sim_param,d_dig_waveform,e_dig_waveform,f_dig_waveform,def_baseline)
   ghi_trigger_flag, ghi_max_sum,ghi_angle,d1 = sum_correlate(sim_param,g_dig_waveform,h_dig_waveform,i_dig_waveform,ghi_baseline)
#########################################


   return abc_max_sum,def_max_sum,ghi_max_sum,abc_angle,def_angle,ghi_angle

if __name__ == '__main__':
   import time
   import matplotlib.pyplot as plt
   
   #print sim_param.num_bits
   """
   # Declare global parameters (in 'if' statement so I can colapse it)
   if(True):
      global TISC_sample_length
      global sample_frequency
      global sim_param.num_samples
      global impulse_position
      global sim_param.num_bits
      global sim_param.draw_flag
      global sim_param.debug
      global b_input_delay
      global c_input_delay
      global SNR
      global sim_param.noise_sigma
      global boresight
      global baseline
      global sim_param.upsample
      global cw_flag
      global cw_amplitude
      global cw_frequency
      global modulation_frequency
      global delay_type_flag
      global sim_param.digitization_factor
      global average_subtract_flag
      global sim_param.window_length
      global sim_param.window_weight
      global six_phi_sector_add
      global sim_param.output_dir
   
   # Assign parameters
   
   # TISC settings parameters
   TISC_sample_length = 16
   sample_frequency = 2600000000.0
   sim_param.num_bits = 3
   sim_param.digitization_factor=32.0
   sim_param.window_length = 8
   sim_param.window_weight = 0.5
   
   # Impulse parameters
   SNR = 5.0
   sim_param.num_samples = 80
   impulse_position = 0
   b_input_delay = -15 # These delays should be negative, since A is the 
   c_input_delay = -17 # top antenna and we expect an upgoing signal
   boresight=0
   baseline = 0
   sim_param.upsample = 1
   
   # Thermal & CW noise parameters
   sim_param.noise_sigma = 32.0
   cw_flag = False
   cw_amplitude = 1.0*sim_param.noise_sigma
   cw_frequency = 260000000.0
   modulation_frequency = 1.0
   
   # Simulation Control
   sim_param.draw_flag = False
   sim_param.debug = False
   delay_type_flag = 1
   average_subtract_flag = False
   six_phi_sector_add = False
   
   # Output control
   sim_param.output_dir = "output/"
   """
   
   threshold = 100
   #num_delays = [63,46]
   if(sim_param.boresight==0):
      if(sim_param.baseline==0):
         abc_baseline = 0
         def_baseline = 1
         ghi_baseline = 1
      elif(sim_param.baseline==1):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 0
   elif(sim_param.boresight==1):
      if(sim_param.baseline==0):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 1
      elif(sim_param.baseline==1):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 0

   SNR = sim_param.SNR
   print "SNR: "+str(SNR)
   average_max_sum=0
   #print "Threshold: "+str(threshold)
                     
   #print 'Correlation test: ' +str(abc_correlation_mean)
   for i in range(0,100):

      abc_max_sum, def_max_sum, ghi_max_sum,abc_angle, def_angle, ghi_angle = TISC_sim(sim_param,SNR,seed=i)                        
      average_max_sum+=abc_max_sum
   average_max_sum/=100
   print "Average max sum: "+str(average_max_sum)
   print abc_max_sum
   print def_max_sum
   print ghi_max_sum

   
   print abc_angle,def_angle,ghi_angle
      
