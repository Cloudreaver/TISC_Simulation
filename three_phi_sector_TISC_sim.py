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
from sum_correlator import sum_correlate
from cw import generate_cw
#import matplotlib.pyplot as plt

def TISC_sim(SNR,threshold,
             b_input_delay,c_input_delay,num_bits=3,
             noise_sigma=32.0,
             sample_freq=2600000000.0,TISC_sample_length=16,
             num_samples=80,upsample=10,cw_flag=0,
             cw_amplitude=20.0,carrier_frequency=260000000.0,modulation_frequency=1.0,
             seed=5522684,draw_flag=0,digitization_factor=32.0,
             delay_type_flag=1,
             output_dir="output/",average_subtract_flag=0,abc_correlation_mean=np.zeros(46),
             def_correlation_mean=np.zeros(44),ghi_correlation_mean=np.zeros(46),trial_run_number=1,boresight=0,baseline=0,six_phi_sector_add=False,window_length=0,window_weight=0.5, debug=False):
   
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
      ghi_impulse_amp = 0.776
      if(baseline==0):
         abc_baseline = 0
         def_baseline = 1
         ghi_baseline = 1
      elif(baseline==1):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 0
   elif(boresight==1):
      abc_impulse_amp = 0.962
      def_impulse_amp = 0.885
      ghi_impulse_amp = 0.650
      if(baseline==0):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 1
      elif(baseline==1):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 0
   
   # Fill numpy arrays with zeros
   a_input_noise = np.zeros(num_samples)
   b_input_noise = np.zeros(num_samples)
   c_input_noise = np.zeros(num_samples)
   d_input_noise = np.zeros(num_samples)
   e_input_noise = np.zeros(num_samples)
   f_input_noise = np.zeros(num_samples)
   g_input_noise = np.zeros(num_samples)
   h_input_noise = np.zeros(num_samples)
   i_input_noise = np.zeros(num_samples)
   time = np.zeros(num_samples)
   upsampled_time = np.zeros(num_samples)
   a_input_signal = np.zeros(num_samples)
   b_input_signal = np.zeros(num_samples)
   c_input_signal = np.zeros(num_samples)
   d_input_signal = np.zeros(num_samples)
   e_input_signal = np.zeros(num_samples)
   f_input_signal = np.zeros(num_samples)
   g_input_signal = np.zeros(num_samples)
   h_input_signal = np.zeros(num_samples)
   i_input_signal = np.zeros(num_samples)
   a_input_signal_noise = np.zeros(num_samples)
   b_input_signal_noise = np.zeros(num_samples)
   c_input_signal_noise = np.zeros(num_samples)
   d_input_signal_noise = np.zeros(num_samples)
   e_input_signal_noise = np.zeros(num_samples)
   f_input_signal_noise = np.zeros(num_samples)
   g_input_signal_noise = np.zeros(num_samples)
   h_input_signal_noise = np.zeros(num_samples)
   i_input_signal_noise = np.zeros(num_samples)
   a_dig_waveform = np.zeros(num_samples)
   b_dig_waveform = np.zeros(num_samples)
   c_dig_waveform = np.zeros(num_samples)
   d_dig_waveform = np.zeros(num_samples)
   e_dig_waveform = np.zeros(num_samples)
   f_dig_waveform = np.zeros(num_samples)
   g_dig_waveform = np.zeros(num_samples)
   h_dig_waveform = np.zeros(num_samples)
   i_dig_waveform = np.zeros(num_samples)
   empty_list = np.zeros(num_samples)


###################################
   # Generate Thermal Noise
   a_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed)
   b_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+1)
   c_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+2)
   d_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+3)
   e_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+4)
   f_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+5)
   g_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+6)
   h_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+7)
   i_input_noise = generate_noise(num_samples,noise_sigma,filter_flag,seed=seed+8)
###################################


#####################################
   # Determine RMS of noise and signal amplitude

   #noise_rms = np.sqrt(np.mean((a_input_noise-noise_mean)**2,))
   
   signal_amp = SNR*2*noise_sigma

#####################################
   
   
#################################
   #Generate CW noise if desired
   if cw_flag:
      cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,cw_amplitude,filter_flag)
      a_input_noise = np.add(a_input_noise,cw_noise)#generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      #a_input_noise += a_input_cw_noise

      # b_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      b_input_noise = np.add(cw_noise,b_input_noise)

      #c_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      c_input_noise = np.add(cw_noise,c_input_noise)
      
      #d_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      d_input_noise = np.add(cw_noise,d_input_noise)

      #e_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      e_input_noise = np.add(cw_noise,e_input_noise)

      #f_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      f_input_noise = np.add(cw_noise,f_input_noise)
      
      #g_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      g_input_noise = np.add(cw_noise,g_input_noise)

      #h_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      h_input_noise = np.add(cw_noise,h_input_noise)

      #i_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      i_input_noise = np.add(cw_noise,i_input_noise)

#####################################
      
   a_input_noise = butter_bandpass_filter(a_input_noise)
   b_input_noise = butter_bandpass_filter(b_input_noise)
   c_input_noise = butter_bandpass_filter(c_input_noise)
   d_input_noise = butter_bandpass_filter(d_input_noise)
   e_input_noise = butter_bandpass_filter(e_input_noise)
   f_input_noise = butter_bandpass_filter(f_input_noise)
   g_input_noise = butter_bandpass_filter(g_input_noise)
   h_input_noise = butter_bandpass_filter(h_input_noise)
   i_input_noise = butter_bandpass_filter(i_input_noise)
 
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
      
      g_input_signal =a_input_signal*ghi_impulse_amp
      h_input_signal =b_input_signal*ghi_impulse_amp
      i_input_signal =c_input_signal*ghi_impulse_amp
      
      """
      if(boresight==0):
         d_input_signal = a_input_signal*0.776 # Average dB loss at -22.5 degrees
         e_input_signal = b_input_signal*0.776 # from Seavey measurements
         f_input_signal = c_input_signal*0.776
         
         g_input_signal = a_input_signal*0.835 # Average dB loss at +22.5 degrees
         h_input_signal = b_input_signal*0.835 # from Seavey measurements
         i_input_signal = c_input_signal*0.835
      elif(boresight==1):
         # For event between two phi sectors, the two antennas are down by about -0.5dB at
         a_input_signal = a_input_signal*0.885 # Average dB los at +11.25 degrees
         b_input_signal = b_input_signal*0.885 # from Seavey measurements
         c_input_signal = c_input_signal*0.885

         d_input_signal = a_input_signal*0.962 # Average dB loss at -11.25 degrees
         e_input_signal = b_input_signal*0.962 # from Seavey measurements
         f_input_signal = c_input_signal*0.962
         
         g_input_signal = a_input_signal*0.650 # Average dB loss at +-33.75 degrees
         h_input_signal = b_input_signal*0.650 # from Seavey measurements
         i_input_signal = c_input_signal*0.650
      """
      if(debug):
         time = np.linspace(0.0,timestep*num_samples,num_samples)
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
      
      if(debug):
         time = np.linspace(0.0,timestep*num_samples,num_samples)
         #plt.plot(time,a_input_signal_noise,time,b_input_signal_noise,time,c_input_signal_noise)
         #plt.plot(time,d_input_signal_noise,time,e_input_signal_noise,time,f_input_signal_noise)
         plt.plot(time,g_input_signal_noise,time,h_input_signal_noise,time,i_input_signal_noise)
         plt.title("impulse+noise")
         plt.show()
      
   else:
      a_input_signal_noise = a_input_noise
      b_input_signal_noise = b_input_noise
      c_input_signal_noise = c_input_noise
      d_input_signal_noise = d_input_noise
      e_input_signal_noise = e_input_noise
      f_input_signal_noise = f_input_noise
      g_input_signal_noise = g_input_noise
      h_input_signal_noise = h_input_noise
      i_input_signal_noise = i_input_noise
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
   g_dig_waveform = digitize(g_input_signal_noise,num_samples,num_bits,digitization_factor)
   h_dig_waveform = digitize(h_input_signal_noise,num_samples,num_bits,digitization_factor)
   i_dig_waveform = digitize(i_input_signal_noise,num_samples,num_bits,digitization_factor)

##########################################
   
   if(debug):
      time = np.linspace(0.0,timestep*num_samples,num_samples)
      #plt.plot(time,a_dig_waveform,time,b_dig_waveform,time,c_dig_waveform)
      #plt.plot(time,d_dig_waveform,time,e_dig_waveform,time,f_dig_waveform)
      plt.plot(time,g_dig_waveform,time,h_dig_waveform,time,i_dig_waveform)
      plt.title("Digitized")
      plt.show()
   
   
##########################################
   # Run the signal through the GLITC module to get trigger
   if(average_subtract_flag):
      
      abc_trigger_flag, abc_max_sum , abc_as_max_sum, abc_correlation_mean, abc_test_sum, abc_as_test_sum,as_abc_angle,abc_angle,d1,d2 = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,threshold,abc_baseline,TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag,correlation_mean=abc_correlation_mean,trial_run_number=trial_run_number)
      def_trigger_flag, def_max_sum , def_as_max_sum, def_correlation_mean, def_test_sum, def_as_test_sum,as_def_angle,def_angle,d1,d2 = sum_correlate(num_samples,d_dig_waveform,e_dig_waveform,f_dig_waveform,threshold,def_baseline,TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag,correlation_mean=def_correlation_mean,trial_run_number=trial_run_number)
      ghi_trigger_flag, ghi_max_sum , ghi_as_max_sum, ghi_correlation_mean, ghi_test_sum, ghi_as_test_sum,as_ghi_angle,ghi_angle,d1,d2 = sum_correlate(num_samples,g_dig_waveform,h_dig_waveform,i_dig_waveform,threshold,ghi_baseline,TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag,correlation_mean=ghi_correlation_mean,trial_run_number=trial_run_number)
      #abc_max_sum
      #print len(a_dig_waveform)  
   else:
      abc_trigger_flag, abc_max_sum,abc_angle,d1 = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,threshold,abc_baseline,TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag,correlation_mean=abc_correlation_mean,trial_run_number=trial_run_number,window_length=window_length,window_weight=window_weight)
      def_trigger_flag, def_max_sum,def_angle,d1 = sum_correlate(num_samples,d_dig_waveform,e_dig_waveform,f_dig_waveform,threshold,def_baseline,TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag,correlation_mean=def_correlation_mean,trial_run_number=trial_run_number,window_length=window_length,window_weight=window_weight)
      ghi_trigger_flag, ghi_max_sum,ghi_angle,d1 = sum_correlate(num_samples,g_dig_waveform,h_dig_waveform,i_dig_waveform,threshold,ghi_baseline,TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag,correlation_mean=ghi_correlation_mean,trial_run_number=trial_run_number,window_length=window_length,window_weight=window_weight)
      #print abc_max_sum
      #print def_max_sum
      #print ghi_max_sum
                                                
   #if(abc_trigger_flag & def_trigger_flag & ghi_trigger_flag):
      #trigger_flag = True
   #else:
      #trigger_flag = False
 

      
   

#########################################

   #dummy = raw_input('Press any key to close')

   if (average_subtract_flag):
      return abc_max_sum,abc_as_max_sum,def_max_sum,def_as_max_sum,ghi_max_sum,ghi_as_max_sum,abc_correlation_mean, def_correlation_mean, ghi_correlation_mean,as_abc_angle,abc_angle,as_def_angle,def_angle,as_ghi_angle,ghi_angle
   else:
      return abc_max_sum,def_max_sum,ghi_max_sum,abc_angle,def_angle,ghi_angle

if __name__ == '__main__':
   #import ROOT
   import time
   import matplotlib.pyplot as plt
   num_samples = 80
   impulse_position = 0
   num_bits = 3
   #signal_amp = 10.0
   draw_flag = False
   debug = False
   # These delays should be negative, since A is the top antenna
   # and we expect an upgoing signal
   b_input_delay = -15
   c_input_delay = -17
   SNR = 3.0
   threshold = 100
   noise_sigma = 32.0
   boresight=0
   baseline = 0
   #num_runs = 100
   upsample = 1
   cw_flag = False
   cw_amplitude = 1.0*noise_sigma
   cw_frequency = 260000000.0
   modulation_frequency = 1.0
   delay_type_flag = 1
   digitization_factor=32.0
   average_subtract_flag = 0
   #global correlation_mean
   abc_correlation_mean = np.zeros(63)
   #abc_correlation_mean.fill(100)
   def_correlation_mean = np.zeros(63)
   #def_correlation_mean.fill(100)
   ghi_correlation_mean = np.zeros(63)
   #ghi_correlation_mean.fill(100)
   num_delays = [63,46]
   if(boresight==0):
      if(baseline==0):
         abc_baseline = 0
         def_baseline = 1
         ghi_baseline = 1
      elif(baseline==1):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 0
   elif(boresight==1):
      if(baseline==0):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 1
      elif(baseline==1):
         abc_baseline = 1
         def_baseline = 0
         ghi_baseline = 0
   

   abc_correlation_mean = np.zeros(num_delays[abc_baseline])
   def_correlation_mean = np.zeros(num_delays[def_baseline])
   ghi_correlation_mean = np.zeros(num_delays[ghi_baseline])
   
   print "SNR: "+str(SNR)
   print "Threshold: "+str(threshold)


   # Get impulse signal
   #a_input_signal = np.zeros(num_samples*upsample)
   #a_input_signal = impulse_gen(num_samples,impulse_position,upsample=upsample,draw_flag=draw_flag)
   
   # Get background estimation
   for i in range(0,5):
      if(average_subtract_flag):
         abc_max_sum,abc_as_max_sum,def_max_sum, def_as_max_sum,ghi_max_sum, ghi_as_max_sum, abc_correlation_mean,def_correlation_mean,ghi_correlation_mean,as_abc_angle,abc_angle,as_def_angle,def_angle,as_ghi_angle,ghi_angle = TISC_sim(0.0,threshold,b_input_delay,c_input_delay,num_bits=num_bits,
                            upsample=upsample,num_samples=num_samples,noise_sigma=noise_sigma,
                            cw_flag=cw_flag,cw_amplitude=cw_amplitude,carrier_frequency=cw_frequency,modulation_frequency=modulation_frequency,
                            draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir="output/",delay_type_flag=delay_type_flag,
                            average_subtract_flag=average_subtract_flag,trial_run_number=(i+1),
                            abc_correlation_mean=abc_correlation_mean,def_correlation_mean=def_correlation_mean,ghi_correlation_mean=ghi_correlation_mean,seed=i)

                            
   #print 'Correlation test: ' +str(abc_correlation_mean)
   for i in range(0,1):
      if(average_subtract_flag):
         abc_max_sum,abc_as_max_sum,def_max_sum, def_as_max_sum,ghi_max_sum, ghi_as_max_sum, abc_correlation_mean,def_correlation_mean,ghi_correlation_mean,as_abc_angle,abc_angle,as_def_angle,def_angle,as_ghi_angle,ghi_angle = TISC_sim(SNR,threshold,b_input_delay,c_input_delay,num_bits=num_bits,
                            upsample=upsample,num_samples=num_samples,noise_sigma=noise_sigma,
                            cw_flag=cw_flag,cw_amplitude=cw_amplitude,carrier_frequency=cw_frequency,modulation_frequency=modulation_frequency,
                            draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir="output/",delay_type_flag=delay_type_flag,
                            average_subtract_flag=average_subtract_flag,trial_run_number=0,
                            abc_correlation_mean=abc_correlation_mean,def_correlation_mean=def_correlation_mean,ghi_correlation_mean=ghi_correlation_mean,seed=i)
      else:
         abc_max_sum, def_max_sum, ghi_max_sum,abc_angle, def_angle, ghi_angle = TISC_sim(SNR,threshold,b_input_delay,c_input_delay,num_bits=num_bits,
                            upsample=upsample,num_samples=num_samples,noise_sigma=noise_sigma,
                            cw_flag=cw_flag,cw_amplitude=cw_amplitude,carrier_frequency=cw_frequency,modulation_frequency=modulation_frequency,
                            draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir="output/",delay_type_flag=delay_type_flag,
                            average_subtract_flag=average_subtract_flag,trial_run_number=0,
                            abc_correlation_mean=abc_correlation_mean,def_correlation_mean=def_correlation_mean,ghi_correlation_mean=ghi_correlation_mean,seed=i,debug=debug)                        
      
      print abc_max_sum
      print def_max_sum
      print ghi_max_sum
      """
      print abc_as_max_sum
      print def_as_max_sum
      print ghi_as_max_sum
      
      print as_abc_angle,as_def_angle,as_ghi_angle
      """
      print abc_angle,def_angle,ghi_angle
      
