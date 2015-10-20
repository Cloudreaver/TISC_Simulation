#!/usr/bin/env python

# Simulates the output of the GLITC 

from array import array
import numpy as np
import csv
#import matplotlib.pyplot as plt



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

def sum_correlate(num_samples,a,b,c,d,e,f,baseline,threshold,TISC_sample_length=16,debug=False):
   #speed_of_light = 2.99*10**8
   #sample_period = 3.5810**(-10)
   #ab_distance = 1.0
   #ac_horz_distance = 1.17
   #ac_vert_distance = 4.7
   #ac_distance = np.sqrt((ac_horz_distance)**2+(ac_vert_distance)**2)
   #print "entering correlator"
   trigger_flag = 0
   #if (delay_type_flag == 1):
   GLITC_delays = read_GLITC_delays(baseline)
   num_delays = len(GLITC_delays)
   
###############################################
# Do Sum correlation
###############################################
   
   #print correlation_mean
   prev_sum = np.zeros(num_delays)
   this_sum = np.zeros(num_delays)
   total_sum = np.zeros(num_delays)
   temp_sum = np.zeros(num_delays)
   max_total_sum = 0
   add_AB = [0] * TISC_sample_length
   add_ABC = [0] * TISC_sample_length

   best_chunk = 0
   b_lower_index_limit = 0
   c_lower_index_limit = 0
   trigger_flag = 0
   timestep = 1.0/2600000000.0
   t = np.linspace(0,timestep*num_samples,num_samples)
   #print baseline
   #print num_delays
   #if (delay_type_flag == 1):
   square_sum = np.zeros(num_delays)
   previous_square_sum = np.zeros(num_delays)
   #else:
      #square_sum_ABC = np.zeros(200*num_samples)
      #previous_square_sum = np.zeros(200*num_samples)
      #delays = np.zeros((200*num_samples,3))
      #i = 0

   # Use GLITC Firmware Delays
   #if (delay_type_flag == 1):
   #print 'Using GLITC type'
   # Build two 16 sample sums (starting with sample 2)
   #num_chunks = int(len(a_dig_waveform)/TISC_sample_length)
   for chunk in range(0,2):
      previous_square_sum = square_sum
      square_sum= np.zeros(num_delays)

      # Get sum correlation for 16 samples
      for i in range(0,num_delays):
         
         # Determine the starting position for each sample
         # Magic number (37) here is the minimum position of the impulse position
         # To allow for all delays to be calcumated
         a_start_pos = chunk*TISC_sample_length+GLITC_delays[i][0]+37
         b_start_pos = chunk*TISC_sample_length+GLITC_delays[i][1]+37
         c_start_pos = chunk*TISC_sample_length+GLITC_delays[i][2]+37
         d_start_pos = chunk*TISC_sample_length+GLITC_delays[i][0]+37
         e_start_pos = chunk*TISC_sample_length+GLITC_delays[i][1]+37
         f_start_pos = chunk*TISC_sample_length+GLITC_delays[i][2]+37
         
         # Make sure we don't run off the front end of the array
         # These delays will be picked up in the next chunk
         # Probably don't need thing anymore
         if ((a_start_pos < 0) or (b_start_pos < 0 ) or (c_start_pos < 0) or (d_start_pos < 0) or (e_start_pos < 0 ) or (f_start_pos < 0)):
            continue
         
         # Add each sample at given delay
         #print len(a_dig_waveform[a_start_pos:a_start_pos+TISC_sample_length])
         #print len(b_dig_waveform[b_start_pos:b_start_pos+TISC_sample_length])
         add_AB = np.add(a[a_start_pos:a_start_pos+TISC_sample_length],b[b_start_pos:b_start_pos+TISC_sample_length])
         add_ABC = np.add(add_AB,c[c_start_pos:c_start_pos+TISC_sample_length])
         add_ABCD = np.add(add_ABC,d[d_start_pos:d_start_pos+TISC_sample_length])
         add_ABCDE = np.add(add_ABCD,e[e_start_pos:e_start_pos+TISC_sample_length])
         add_ABCDEF = np.add(add_ABCDE,f[f_start_pos:f_start_pos+TISC_sample_length])
         
         
         #print add_ABC
         square = np.square(add_ABCDEF)
         #print square_ABC
         square_sum[i] = np.sum(square)
         #print square_sum_ABC[i]
         if(debug):
            print len(t)
            print len(a)
            print len(b)
            print len(c)
            print "Temp sum and delay"
            print square_sum[i]
            print GLITC_delays[i]
            """
            plt.figure(1)
            plt.plot(t[a_start_pos:a_start_pos+TISC_sample_length],a[a_start_pos:a_start_pos+TISC_sample_length],t[a_start_pos:a_start_pos+TISC_sample_length],b[b_start_pos:b_start_pos+TISC_sample_length],t[a_start_pos:a_start_pos+TISC_sample_length],c[c_start_pos:c_start_pos+TISC_sample_length])
            plt.plot(t[a_start_pos:a_start_pos+TISC_sample_length],d[a_start_pos:a_start_pos+TISC_sample_length],t[a_start_pos:a_start_pos+TISC_sample_length],e[b_start_pos:b_start_pos+TISC_sample_length],t[a_start_pos:a_start_pos+TISC_sample_length],f[c_start_pos:c_start_pos+TISC_sample_length])
            
            plt.figure(2)
            plt.plot(t[a_start_pos:a_start_pos+TISC_sample_length],add_ABCDEF)
            plt.figure(3)
            plt.plot(t[a_start_pos:a_start_pos+TISC_sample_length],square)
            plt.show()
            """
   # Add all the 16 sample sums for each delay
   total_sum = np.add(square_sum,previous_square_sum) 
   
   # Find the maximum sum
   max_total_sum = np.amax(total_sum)
   
   # Find the delays for the largest sum
   best_delays = GLITC_delays[np.argmax(total_sum)]
   if(debug):
      #print "Best sum and delays"
      #print max_total_sum
      print best_delays
   
   #print "Max Sum: "+str(max_total_sum)
   #print best_delays
      
   return trigger_flag, max_total_sum,best_delays[3],total_sum


if __name__ == '__main__':
   import numpy as np
   from impulse import impulse_gen
   from digitizer import digitize
   import matplotlib.pyplot as plt
   from noise import generate_noise
   threshold = 300
   num_samples = 80
   baseline = 0
   upsample = 10
   num_bits = 3
   noise_sigma = 32.0
   SNR = 5.0
   num_upsamples = num_samples*upsample
   a_dig_waveform = np.zeros(num_upsamples)
   a_waveform = np.zeros(num_samples)
   #a_dig_waveform[20] = 3.5
   #a_dig_waveform[21] = -3.5
   #a_dig_waveform[22] = 2.5
   #a_dig_waveform[23] = -2.5
   a_delay = 0
   b_delay = -15
   c_delay = -17
   TISC_sample_length = 16
   delay_type_flag = 1
   average_subtract_flag = 1
   correlation_mean = np.zeros(63)
   correlation_mean.fill(50)
   filter_flag = 0
   debug = False
   timestep = 1.0/2600000000.0
   t = np.linspace(0,timestep*num_samples,num_samples)
   signal_amp = SNR*2*noise_sigma

   for i in range(0,1):
      a_waveform = impulse_gen(num_samples,a_delay,upsample,draw_flag=0,output_dir='output/')
      empty_list = np.zeros(num_samples)
      difference=np.amax(a_waveform)-np.amin(a_waveform) # Get peak to peak voltage
      a_waveform *= (1/difference) # Normalize input
      a_waveform *= signal_amp # Amplify
      a_imp_dig_wfm = digitize(a_waveform,num_samples,num_bits,digitization_factor=noise_sigma)
      a_waveform = np.add(a_waveform,generate_noise(num_samples,noise_sigma,filter_flag))
      b_waveform = np.concatenate([a_waveform[-b_delay:],empty_list[:(-1)*b_delay]])
      c_waveform = np.concatenate([a_waveform[-c_delay:],empty_list[:(-1)*c_delay]])
      b_waveform = np.add(b_waveform,generate_noise(num_samples,noise_sigma,filter_flag))
      c_waveform = np.add(c_waveform,generate_noise(num_samples,noise_sigma,filter_flag))
      
      d_waveform = np.add(a_waveform,generate_noise(num_samples,noise_sigma,filter_flag))
      e_waveform = np.concatenate([d_waveform[-b_delay:],empty_list[:(-1)*b_delay]])
      f_waveform = np.concatenate([d_waveform[-c_delay:],empty_list[:(-1)*c_delay]])
      e_waveform = np.add(b_waveform,generate_noise(num_samples,noise_sigma,filter_flag))
      f_waveform = np.add(c_waveform,generate_noise(num_samples,noise_sigma,filter_flag))
      #a_waveform[40]=3.5
      #a_waveform[41]=-3.5
      
      
      
      
      a_dig_waveform = digitize(a_waveform,num_samples,num_bits,digitization_factor=noise_sigma)
      b_dig_waveform = digitize(b_waveform,num_samples,num_bits,digitization_factor=noise_sigma)
      c_dig_waveform = digitize(c_waveform,num_samples,num_bits,digitization_factor=noise_sigma)
      
      
      start =37
      end = start+32

      #print np.sum(np.square(3*a_dig_waveform[start:end]))
      #plt.plot(t[start:end],a_dig_waveform[start:end],t[start:end],b_dig_waveform[start:end],t[start:end],c_dig_waveform[start:end])
      plt.plot(t,a_dig_waveform,t,b_dig_waveform,t,c_dig_waveform)
      plt.show()
      #print a_dig_waveform
      #print i
      passed_flag, max_sum, as_max_sum, correlation_mean, dummy1,dummy2,max_sum_angle, as_max_sum_angle = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,
                                                threshold,baseline,TISC_sample_length=TISC_sample_length,delay_type_flag=delay_type_flag,
                                                average_subtract_flag=average_subtract_flag, correlation_mean=correlation_mean,trial_run_number=0,debug=debug)
   print passed_flag, max_sum, as_max_sum, max_sum_angle, as_max_sum_angle
