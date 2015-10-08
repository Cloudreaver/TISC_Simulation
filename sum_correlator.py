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

def sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,threshold,baseline,TISC_sample_length=16,delay_type_flag=0,average_subtract_flag=0,correlation_mean=np.zeros(44),trial_run_number=0,debug=False):
   #speed_of_light = 2.99*10**8
   #sample_period = 3.5810**(-10)
   #ab_distance = 1.0
   #ac_horz_distance = 1.17
   #ac_vert_distance = 4.7
   #ac_distance = np.sqrt((ac_horz_distance)**2+(ac_vert_distance)**2)
   #print "entering correlator"
   trigger_flag = 0
   if (delay_type_flag == 1):
      GLITC_delays = read_GLITC_delays(baseline)
      num_delays = len(GLITC_delays)
      """
      if(baseline==0):
         # These delays are for the lower A antenna
         read_GLITC_
         GLITC_delays = np.array([[0,-7,-5],
                           [0,-7,-6],
                           [0,-8,-7],
                           [0,-8,-8],
                           [0,-8,-9],
                           [0,-9,-8],
                           [0,-9,-9],
                           [0,-10,-9],
                           [0,-10,-10],
                           [0,-11,-11],
                           [0,-11,-12],
                           [0,-12,-12],
                           [0,-12,-13],
                           [0,-13,-13],
                           [0,-13,-14],
                           [0,-13,-15],
                           [0,-14,-15],
                           [0,-14,-16],
                           [0,-15,-16],
                           [0,-15,-17],
                           [0,-16,-18],
                           [0,-16,-19],
                           [0,-17,-19],
                           [0,-18,-20],
                           [0,-18,-21],
                           [0,-18,-22],
                           [0,-19,-22],
                           [0,-19,-23],
                           [0,-20,-23],
                           [0,-20,-24],
                           [0,-20,-25],
                           [0,-21,-25],
                           [0,-21,-26],
                           [0,-22,-26],
                           [0,-22,-27],
                           [0,-22,-28],
                           [0,-23,-28],
                           [0,-23,-29],
                           [0,-23,-30],
                           [0,-24,-30],
                           [0,-24,-30],
                           [0,-24,-31],
                           [0,-24,-32],
                           [0,-25,-32]
                           ])
         
      #GLITC_delays = np.array([[0,0,0],[1,0,0]])
      """
   
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
   if (delay_type_flag == 1):
      square_sum_ABC = np.zeros(num_delays)
      previous_square_sum = np.zeros(num_delays)
   else:
      square_sum_ABC = np.zeros(200*num_samples)
      previous_square_sum = np.zeros(200*num_samples)
      delays = np.zeros((200*num_samples,3))
      i = 0

   # Use GLITC Firmware Delays
   if (delay_type_flag == 1):
      #print 'Using GLITC type'
      # Build two 16 sample sums (starting with sample 2)
      #num_chunks = int(len(a_dig_waveform)/TISC_sample_length)
      for chunk in range(0,2):
         previous_square_sum = square_sum_ABC
         square_sum_ABC= np.zeros(num_delays)

         # Get sum correlation for 16 samples
         for i in range(0,num_delays):
            
            # Determine the starting position for each sample
            # Magic number (37) here is the minimum position of the impulse position
            # To allow for all delays to be calcumated
            a_start_pos = chunk*TISC_sample_length+GLITC_delays[i][0]+37
            b_start_pos = chunk*TISC_sample_length+GLITC_delays[i][1]+37
            c_start_pos = chunk*TISC_sample_length+GLITC_delays[i][2]+37
            
            # Make sure we don't run off the front end of the array
            # These delays will be picked up in the next chunk
            # Probably don't need thing anymore
            if ((a_start_pos < 0) or (b_start_pos < 0 ) or (c_start_pos < 0)):
               continue
            
            # Add each sample at given delay
            #print len(a_dig_waveform[a_start_pos:a_start_pos+TISC_sample_length])
            #print len(b_dig_waveform[b_start_pos:b_start_pos+TISC_sample_length])
            add_AB = np.add(a_dig_waveform[a_start_pos:a_start_pos+TISC_sample_length],b_dig_waveform[b_start_pos:b_start_pos+TISC_sample_length])
            #print add_AB
            add_ABC = np.add(add_AB,c_dig_waveform[c_start_pos:c_start_pos+TISC_sample_length])
            #print add_ABC
            square_ABC = np.square(add_ABC)
            #print square_ABC
            square_sum_ABC[i] = np.sum(square_ABC)
            #print square_sum_ABC[i]
            #if(debug):
               #print len(t)
               #print len(a_dig_waveform)
               #print len(b_dig_waveform)
               #print len(c_dig_waveform)
               #print "Temp sum and delay"
               #print square_sum_ABC[i]
               #print GLITC_delays[i]
               #plt.figure(1)
               #plt.plot(t[a_start_pos:a_start_pos+TISC_sample_length],a_dig_waveform[a_start_pos:a_start_pos+TISC_sample_length],t[a_start_pos:a_start_pos+TISC_sample_length],b_dig_waveform[b_start_pos:b_start_pos+TISC_sample_length],t[a_start_pos:a_start_pos+TISC_sample_length],c_dig_waveform[c_start_pos:c_start_pos+TISC_sample_length])
               #plt.figure(2)
               #plt.plot(t[a_start_pos:a_start_pos+TISC_sample_length],add_ABC)
               #plt.figure(3)
               #plt.plot(t[a_start_pos:a_start_pos+TISC_sample_length],square_ABC)
               #plt.show()
            
      # Add all the 16 sample sums for each delay
      total_sum = np.add(square_sum_ABC,previous_square_sum) 
      
      # Find the maximum sum
      max_total_sum = np.amax(total_sum)
      
      # Find the delays for the largest sum
      best_delays = GLITC_delays[np.argmax(total_sum)]
      if(debug):
         #print "Best sum and delays"
         #print max_total_sum
         print best_delays
      if(average_subtract_flag):
         # Average over sums during the trial period
         if (trial_run_number>0):
            #print correlation_mean
            #print len(total_sum)
            #print len(correlation_mean)
            correlation_mean = (1.0/float(trial_run_number))*(((float(trial_run_number)-1.0)*correlation_mean)+total_sum)
            
         # After the trial period (trial_run_number=0) actually subtracted the average
         #print len(total_sum)
         #print len(correlation_mean)
         as_total_sum = np.subtract(total_sum,correlation_mean)
         as_max_total_sum = np.amax(as_total_sum)
         as_best_delays = GLITC_delays[np.argmax(as_total_sum)]
         #print trial_run_number
         #print 'Total Sum: '+str(total_sum)
         #print 'Correlation mean in correlator'+str(correlation_mean)
         #print "Total Sum: "+str(total_sum[0])
         #print "Running Mean: "+str(correlation_mean[0])
         #print ((float(trial_run_number)-1)*float(correlation_mean[0]))
         #print as_total_sum[0]
         #print "\n"
      #print trial_run_number
      #print correlation_mean[10]
      #print total_sum
      #print "\n"
      #print total_sum
      #print len(total_sum)
      #print len(GLITC_delays)
   #print "Max Sum: "+str(max_total_sum)
   #print best_delays
   """
   # All possible delays (may not work anymore)
   else:
   
      #Set the lower index limit of Ch B/C
      for chunk in range(0,(num_samples/TISC_sample_length)):
         b_lower_index_limit = 0
         c_lower_index_limit = 0
         previous_sum_ABC = square_sum_ABC

         # Shift Ch B/C around
         for b in range(b_lower_index_limit,len(a_dig_waveform)-TISC_sample_length):
            for c in range(c_lower_index_limit,len(a_dig_waveform)-TISC_sample_length):
               this_sum = 0
               add_AB = np.add(a_dig_waveform[chunk*TISC_sample_length:(chunk*TISC_sample_length)+TISC_sample_length],b_dig_waveform[b:TISC_sample_length+b])
               add_ABC = np.add(add_AB,c_dig_waveform[c:TISC_sample_length+c])
               square_ABC = add_ABC**2
               square_sum_ABC[i] = np.sum(square_ABC)
               delays[i] = [chunk*TISC_sample_length,b,c]
               i += 1
               #print "a: "+str(chunk*TISC_sample_length)+":"+str((chunk*TISC_sample_length)+TISC_sample_length)+"\tb: "+str(b)+":"+str(b+TISC_sample_length)+"\tc: "+str(c)+":"+str(c+TISC_sample_length)
                  
      total_sum = np.add(square_sum_ABC,previous_square_sum)
      max_total_sum = np.amax(total_sum)
      best_delays = delays[np.argmax(total_sum)]
      
      #print total_sum
      #print "Max sum: " +str(max_total_sum)
      #print best_delays

   #if(max_total_sum > threshold):
      #trigger_flag = 1
      #print "Event passes"
      
  #print "Threshold: " + str(threshold)
   #print "max_total_sum: " + str(max_total_sum)
   #print "A Chunk: " + str(best_chunk)
   #print "B Delay: " + str(best_b_delay)#+'\t'+str(b_input_delay-best_b_delay)
   #print "C Delay: " + str(best_c_delay)#+'\t'+str(c_input_delay-best_c_delay)
   #print "\n\n"
   """
   """
   if (average_subtract_flag):
      return trigger_flag, max_total_sum, as_max_total_sum, correlation_mean
   else:
      return trigger_flag, max_total_sum
   """
      
   if (average_subtract_flag):
      return trigger_flag, max_total_sum, as_max_total_sum, correlation_mean, total_sum[0], as_total_sum[0],best_delays[3],as_best_delays[3]
   else:
      return trigger_flag, max_total_sum,best_delays[3]


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
