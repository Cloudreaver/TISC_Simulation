#!/usr/bin/env python

# A very simple impulsive signal + noise generator to get the 

import random
#from ROOT import TTree, TBranch, gROOT,gSystem,TImage, TStyle, TFile, gDirectory, TH1D, TCanvas, gStyle, TGraph
from array import array
import numpy as np
from impulse import impulse_gen
from anita_filter import butter_bandpass, butter_bandpass_filter
from noise import generate_noise
from digitizer import digitize
from six_antenna_sum_correlator import sum_correlate
from cw import generate_cw
#import matplotlib.pyplot as plt

def TISC_sim(SNR,threshold,
             b_input_delay,c_input_delay,num_bits=3,
             noise_sigma=32.0,
             sample_freq=2600000000.0,TISC_sample_length=16,
             num_samples=80,upsample=10,
             seed=5522684,draw_flag=0,digitization_factor=32.0,
             output_dir="output/",boresight=0,baseline=0,debug=False):
   
   # Setup
   save_output_flag = 0
   #if(save_output_flag):
      #outfile = str(output_dir+"/test.root")
   trigger_flag = 0
   #num_bits = 3 # Number of bits available to the digitizer
   #num_samples = num_samples*upsample
   filter_flag = False
   #sample_frequency = 2800000000.0
   def_max_sum = 0.0
   def_as_max_sum = 0.0
   ghi_max_sum = 0.0
   ghi_as_max_sum = 0.0
   timestep = 1.0/sample_freq
   
   
   
   # Phi sectors have alternating baselines
   if(boresight==0):
      abc_impulse_amp = 1.000
      def_impulse_amp = 0.835
   elif(boresight==1):
      abc_impulse_amp = 0.962
      def_impulse_amp = 0.885
   
   # Fill numpy arrays with zeros
   a_input_noise = np.zeros(num_samples)
   b_input_noise = np.zeros(num_samples)
   c_input_noise = np.zeros(num_samples)
   d_input_noise = np.zeros(num_samples)
   e_input_noise = np.zeros(num_samples)
   f_input_noise = np.zeros(num_samples)

   time = np.zeros(num_samples)
   upsampled_time = np.zeros(num_samples)
   a_input_signal = np.zeros(num_samples)
   b_input_signal = np.zeros(num_samples)
   c_input_signal = np.zeros(num_samples)
   d_input_signal = np.zeros(num_samples)
   e_input_signal = np.zeros(num_samples)
   f_input_signal = np.zeros(num_samples)

   a_input_signal_noise = np.zeros(num_samples)
   b_input_signal_noise = np.zeros(num_samples)
   c_input_signal_noise = np.zeros(num_samples)
   d_input_signal_noise = np.zeros(num_samples)
   e_input_signal_noise = np.zeros(num_samples)
   f_input_signal_noise = np.zeros(num_samples)

   a_dig_waveform = np.zeros(num_samples)
   b_dig_waveform = np.zeros(num_samples)
   c_dig_waveform = np.zeros(num_samples)
   d_dig_waveform = np.zeros(num_samples)
   e_dig_waveform = np.zeros(num_samples)
   f_dig_waveform = np.zeros(num_samples)
   
   empty_list = np.zeros(num_samples)


###################################
   # Generate Thermal Noise
   a_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+0)
   b_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+1)
   c_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+2)
   
   d_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+3)
   e_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+4)
   f_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+5)
###################################


#####################################
   # Determine RMS of noise and signal amplitude

   #noise_rms = np.sqrt(np.mean((a_input_noise-noise_mean)**2,))
   
   signal_amp = SNR*2*noise_sigma

#####################################
      
   a_input_noise = butter_bandpass_filter(a_input_noise)
   b_input_noise = butter_bandpass_filter(b_input_noise)
   c_input_noise = butter_bandpass_filter(c_input_noise)
   
   d_input_noise = butter_bandpass_filter(d_input_noise)
   e_input_noise = butter_bandpass_filter(e_input_noise)
   f_input_noise = butter_bandpass_filter(f_input_noise)

 
#####################################
   # Generate impulse
   if (SNR != 0):
      # Generate Signal and Amplify
      a_input_signal = impulse_gen(num_samples,upsample,draw_flag=draw_flag,output_dir=output_dir)
      difference=np.amax(a_input_signal)-np.amin(a_input_signal) # Get peak to peak voltage
      a_input_signal *= (1/difference) # Normalize input
      a_input_signal *= signal_amp # Amplify
      #b_input_signal = np.concatenate([a_input_signal[:num_samples+b_input_delay],empty_list[:(-1)*b_input_delay]])
      #c_input_signal = np.concatenate([a_input_signal[:num_samples+c_input_delay],empty_list[:(-1)*c_input_delay]])
      b_input_signal = np.concatenate([a_input_signal[-b_input_delay:],empty_list[:(-1)*b_input_delay]])
      c_input_signal = np.concatenate([a_input_signal[-c_input_delay:],empty_list[:(-1)*c_input_delay]])
      
      
      
      a_input_signal =a_input_signal*abc_impulse_amp
      b_input_signal =b_input_signal*abc_impulse_amp
      c_input_signal =c_input_signal*abc_impulse_amp
      
      d_input_signal =a_input_signal*def_impulse_amp
      e_input_signal =b_input_signal*def_impulse_amp
      f_input_signal =c_input_signal*def_impulse_amp
      
      """
      time = np.linspace(0.0,timestep*num_samples,num_samples)
      plt.plot(time,a_input_signal,time,b_input_signal,time,c_input_signal)
      plt.plot(time,d_input_signal,time,e_input_signal,time,f_input_signal)
      plt.plot(time,g_input_signal,time,h_input_signal,time,i_input_signal)
      plt.title("impulse")
      plt.show()  
      """
      # Add the signal to the noise
      a_input_signal_noise = np.add(a_input_noise, a_input_signal)
      b_input_signal_noise = np.add(b_input_noise, b_input_signal)
      c_input_signal_noise = np.add(c_input_noise, c_input_signal)
      
      d_input_signal_noise = np.add(d_input_noise, d_input_signal)
      e_input_signal_noise = np.add(e_input_noise, e_input_signal)
      f_input_signal_noise = np.add(f_input_noise, f_input_signal)

      """
      time = np.linspace(0.0,timestep*num_samples,num_samples)
      plt.plot(time,a_input_signal_noise,time,b_input_signal_noise,time,c_input_signal_noise)
      plt.plot(time,d_input_signal_noise,time,e_input_signal_noise,time,f_input_signal_noise)
      #plt.plot(time,g_input_signal_noise,time,h_input_signal_noise,time,i_input_signal_noise)
      plt.title("impulse+noise")
      plt.show()
      """
   else:
      a_input_signal_noise = a_input_noise
      b_input_signal_noise = b_input_noise
      c_input_signal_noise = c_input_noise
      
      d_input_signal_noise = d_input_noise
      e_input_signal_noise = e_input_noise
      f_input_signal_noise = f_input_noise

##########################################
   #time = np.linspace(0.0,timestep*num_samples,num_samples)
   #plt.plot(time,a_input_noise,time,b_input_noise,time,c_input_noise)
   #plt.title("noise")
   #plt.show()
##########################################
   # Digitized the incoming signal and noise (RITC)
   a_dig_waveform = digitize(a_input_signal_noise,num_samples,num_bits,digitization_factor)
   b_dig_waveform = digitize(b_input_signal_noise,num_samples,num_bits,digitization_factor)
   c_dig_waveform = digitize(c_input_signal_noise,num_samples,num_bits,digitization_factor)
   
   d_dig_waveform = digitize(d_input_signal_noise,num_samples,num_bits,digitization_factor)
   e_dig_waveform = digitize(e_input_signal_noise,num_samples,num_bits,digitization_factor)
   f_dig_waveform = digitize(f_input_signal_noise,num_samples,num_bits,digitization_factor)


##########################################
   """
   time = np.linspace(0.0,timestep*num_samples,num_samples)
   plt.plot(time,a_dig_waveform,time,b_dig_waveform,time,c_dig_waveform)
   plt.plot(time,d_dig_waveform,time,e_dig_waveform,time,f_dig_waveform)
   #plt.plot(time,g_dig_waveform,time,h_dig_waveform,time,i_dig_waveform)
   plt.title("Digitized")
   plt.show()
   """
   
##########################################
   # Run the signal through the GLITC module to get trigger

   trigger_flag, max_total_sum,best_angle,total_sum = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,d_dig_waveform,e_dig_waveform,f_dig_waveform,baseline,threshold,TISC_sample_length,debug=debug)
   #print abc_max_sum
   #print def_max_sum
   #print ghi_max_sum
                                             

 

      
   

#########################################

   #dummy = raw_input('Press any key to close')


   return max_total_sum,best_angle
   
if __name__ == '__main__':
   #import ROOT
   import time
   import matplotlib.pyplot as plt
   num_samples = 80
   impulse_position = 0
   num_bits = 3
   #signal_amp = 10.0
   draw_flag = False
   # These delays should be negative, since A is the top antenna
   # and we expect an upgoing signal
   b_input_delay = -15
   c_input_delay = -17
   SNR = 5.0
   threshold = 100
   noise_sigma = 32.0
   boresight=0
   baseline = 0
   #num_runs = 100
   upsample = 1
   delay_type_flag = 1
   digitization_factor=32.0


   

   
   print "SNR: "+str(SNR)
   print "Threshold: "+str(threshold)


   # Get impulse signal
   #a_input_signal = np.zeros(num_samples*upsample)
   #a_input_signal = impulse_gen(num_samples,impulse_position,upsample=upsample,draw_flag=draw_flag)
   

   total_sum, best_angle = TISC_sim(SNR,100,
						b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
						noise_sigma=noise_sigma,
                  digitization_factor=digitization_factor,boresight=boresight,baseline=baseline,debug=True)
   print total_sum, best_angle
