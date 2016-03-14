#!/usr/bin/env python

# Simulates the output of the GLITC 

from array import array
import numpy as np
import csv


def read_GLITC_delays(baseline):
   if(baseline==0):
      filename = "GLITC_long_baseline_delays.dat"
   elif(baseline==1):
      filename = "GLITC_short_baseline_delays.dat"
      
   angles = []
   a_delay = []
   b_delay = []
   c_delay = []
      
   with open(filename,'rb') as datafile:
      datareader = csv.reader(datafile, delimiter=',', quotechar='|')

      row_number = 0
      header_flag = 0

      # Read in and plot the data
      for row in datareader:
         if (header_flag == 0):
            header_flag = 1
         else:			
            angles.append(float(row[0]))
            a_delay.append(float(row[1]))
            b_delay.append(float(row[2]))
            c_delay.append(float(row[3]))
            
      tmp_GLITC_delays = []
      tmp_GLITC_delays.append(a_delay)
      tmp_GLITC_delays.append(b_delay)
      tmp_GLITC_delays.append(c_delay)
      tmp_GLITC_delays.append(angles)
      
      GLITC_delays = np.transpose(np.array(tmp_GLITC_delays))
      
   return GLITC_delays

def sum_correlate(sim_param,a_dig_waveform,b_dig_waveform,c_dig_waveform,baseline):

   if(sim_param.debug): import matplotlib.pyplot as plt
   trigger_flag = 0
   if (sim_param.delay_type_flag == 1):
      GLITC_delays = read_GLITC_delays(baseline)
      num_delays = len(GLITC_delays)
      #print num_delays
      
   trigger_flag = 0
   timestep = 1.0/sim_param.sample_frequency
   
   total_sum = np.zeros(num_delays)
   max_total_sum = 0
   prev_add_AB = np.zeros(sim_param.TISC_sample_length)
   prev_add_ABC = np.zeros(sim_param.TISC_sample_length)
   add_AB = np.zeros(sim_param.TISC_sample_length)
   add_ABC = np.zeros(sim_param.TISC_sample_length)
   t = np.linspace(0,timestep*sim_param.num_samples,sim_param.num_samples)
   square_sum_ABC = np.zeros(num_delays)
   prev_square_sum_ABC = np.zeros(num_delays)
   

   
   
   ###############################################
   # Do Sum correlation
   ###############################################

   # Get sum correlation for 16 samples
   for i in range(0,num_delays):
      
      # Determine the starting position for each sample
      # Magic number (37) here is the minimum position of the impulse position
      # To allow for all delays to be calcumated
      prev_a_start_pos = 37
      prev_a_end_pos = prev_a_start_pos+sim_param.TISC_sample_length
      a_start_pos = prev_a_end_pos+1
      a_end_pos = a_start_pos+sim_param.TISC_sample_length
      
      prev_b_start_pos = int(prev_a_start_pos+GLITC_delays[i][1])
      prev_b_end_pos = prev_b_start_pos+sim_param.TISC_sample_length
      b_start_pos = prev_b_start_pos+1
      b_end_pos = b_start_pos+sim_param.TISC_sample_length
      
      prev_c_start_pos = int(prev_a_start_pos+GLITC_delays[i][2])
      prev_c_end_pos = prev_c_start_pos+sim_param.TISC_sample_length
      c_start_pos = prev_c_start_pos+1
      c_end_pos = c_start_pos+sim_param.TISC_sample_length
      
      # Add each sample at given delay
      #print a_dig_waveform[prev_a_start_pos:prev_a_end_pos]
      #print a_dig_waveform[a_start_pos:a_end_pos]
      prev_add_AB = np.add(a_dig_waveform[prev_a_start_pos:prev_a_end_pos],b_dig_waveform[prev_b_start_pos:prev_b_end_pos])
      add_AB = np.add(a_dig_waveform[a_start_pos:a_end_pos],b_dig_waveform[b_start_pos:b_end_pos])
      
      prev_add_ABC = np.add(prev_add_AB,c_dig_waveform[prev_c_start_pos:prev_c_end_pos])
      add_ABC = np.add(add_AB,c_dig_waveform[c_start_pos:c_end_pos])
      
      # Apply window if desired
      if(sim_param.window_length):
         #print "Windowing"
         #print sim_param.window_length
         #print sim_param.window_weight
         add_ABC[len(add_ABC)-sim_param.window_length:len(add_ABC)] *= sim_param.window_weight
      
      # Square the elements
      prev_square_ABC = np.square(prev_add_ABC)
      square_ABC = np.square(add_ABC)

      # Accumulate the arrays
      prev_square_sum_ABC[i] = np.sum(prev_square_ABC)
      square_sum_ABC[i] = np.sum(square_ABC)
      

   # Add all the 16 sample sums for each delay
   total_sum = np.add(square_sum_ABC,prev_square_sum_ABC) 
   #print prev_square_sum_ABC
   #print square_sum_ABC 
   
   # Find the maximum sum
   max_total_sum = np.amax(total_sum)
   
   # Find the delays for the largest sum
   best_delays = GLITC_delays[np.argmax(total_sum)]
   
   if(sim_param.debug and max_total_sum>400):
      print "Best sum and delays"
      print max_total_sum
      print best_delays
      bplot =np.roll(b_dig_waveform,int(best_delays[1]))
      cplot = np.roll(c_dig_waveform,int(best_delays[2]))
      plt.figure(10)
      plt.plot(t,a_dig_waveform,t,bplot,t,cplot)
      plt.show()
   
   
   #print trial_run_number
   #print correlation_mean[10]
   #print total_sum
   #print "\n"
   #print total_sum
   #print len(total_sum)
   #print len(GLITC_delays)
   #print "Max Sum: "+str(max_total_sum)
   #print best_delays
   
      
  
   return trigger_flag, max_total_sum,best_delays[3],total_sum


if __name__ == '__main__':
   import numpy as np
   from impulse import impulse_gen
   from digitizer import digitize
   import matplotlib.pyplot as plt
   from noise import generate_noise
   from simulation_parameters import sim_param
   
   '''
   threshold = 300
   num_samples = 80
   baseline = 0
   upsample = 10
   num_bits = 3
   noise_sigma = 32.0
   digitization_factor=32.0
   SNR = 1.8
   num_upsamples = num_samples*upsample
   #a_dig_waveform = np.zeros(num_upsamples)
   
   #a_dig_waveform[20] = 3.5
   #a_dig_waveform[21] = -3.5
   #a_dig_waveform[22] = 2.5
   #a_dig_waveform[23] = -2.5
   a_delay = 0
   b_delay = -14
   c_delay = -16
   TISC_sample_length = 16
   delay_type_flag = 1
   average_subtract_flag = 0
   correlation_mean = np.zeros(63)
   correlation_mean.fill(50)
   filter_flag = 0
   debug = False
   
   
   
   #window_length=0
   #window_weight=0.5
   '''
   
   a_waveform = np.zeros(sim_param.num_samples)
   signal_amp = sim_param.SNR*2*sim_param.noise_sigma
   timestep = 1.0/2600000000.0
   t = np.linspace(0,timestep*sim_param.num_samples,sim_param.num_samples)
   average_max_sum=0
   
   for i in range(0,100):
      #a_waveform = impulse_gen(num_samples,a_delay,upsample,draw_flag=0,output_dir='output/')
      a_dig_waveform = np.zeros(sim_param.num_samples)
      b_dig_waveform = np.zeros(sim_param.num_samples)
      c_dig_waveform = np.zeros(sim_param.num_samples)
      a_waveform[40:52]=[3.5,-3.5 ,3.0,-3.0,2.5,-2.5,2.0,-2.0,1.5,-1.5,1.0,-1.0]
      empty_list = np.zeros(sim_param.num_samples)
      difference=np.amax(a_waveform)-np.amin(a_waveform) # Get peak to peak voltage
      a_waveform *= (1/difference) # Normalize input
      a_waveform *= signal_amp # Amplify
      a_imp_dig_wfm = digitize(a_waveform,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
      a_waveform = np.add(a_waveform,generate_noise(sim_param.num_samples,noise_sigma=sim_param.noise_sigma))
      b_waveform = np.concatenate([a_waveform[-sim_param.b_input_delay:],empty_list[:(-1)*sim_param.b_input_delay]])
      c_waveform = np.concatenate([a_waveform[-sim_param.c_input_delay:],empty_list[:(-1)*sim_param.c_input_delay]])
      b_waveform = np.add(b_waveform,generate_noise(sim_param.num_samples,noise_sigma=sim_param.noise_sigma))
      c_waveform = np.add(c_waveform,generate_noise(sim_param.num_samples,noise_sigma=sim_param.noise_sigma))
      
      #a_dig_waveform[40]=3.5
      #b_dig_waveform = np.roll(a_dig_waveform,b_delay)
      #c_dig_waveform = np.roll(a_dig_waveform,c_delay)

      
      
      a_dig_waveform = digitize(a_waveform,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
      b_dig_waveform = digitize(b_waveform,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
      c_dig_waveform = digitize(c_waveform,sim_param.num_samples,sim_param.num_bits,sim_param.digitization_factor)
      
      
      start =37
      end = start+32

      #print np.sum(np.square(3*a_dig_waveform[start:end]))
      #plt.plot(t[start:end],a_dig_waveform[start:end],t[start:end],b_dig_waveform[start:end],t[start:end],c_dig_waveform[start:end])
      #plt.plot(t,a_dig_waveform,t,b_dig_waveform,t,c_dig_waveform)
      #plt.show()
      #print a_dig_waveform
      #print i
      passed_flag, max_sum, best_delays,total_sum = sum_correlate(sim_param,a_dig_waveform,b_dig_waveform,c_dig_waveform,sim_param.baseline)
      #print passed_flag, max_sum, best_delays#, total_sum
      average_max_sum += max_sum
   average_max_sum /=100
   print average_max_sum
