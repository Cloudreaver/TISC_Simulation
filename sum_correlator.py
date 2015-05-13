#!/usr/bin/env python

# Simulates the output of the GLITC 

from array import array
import numpy as np



def sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,threshold,TISC_sample_length=16,delay_type_flag=0):
   import matplotlib.pyplot as plt
   #speed_of_light = 2.99*10**8
   #sample_period = 3.5810**(-10)
   #ab_distance = 1.0
   #ac_horz_distance = 1.17
   #ac_vert_distance = 4.7
   #ac_distance = np.sqrt((ac_horz_distance)**2+(ac_vert_distance)**2)

   trigger_flag = 0
   if (delay_type_flag == 1):
      GLITC_delays = np.array([[0,30,42],
                           [0,29,41],
                           [0,29,40],
                           [0,28,38],
                           [0,27,37],
                           [0,27,36],
                           [0,26,35],
                           [0,25,34],
                           [0,25,33],
                           [0,24,32],
                           [0,23,31],
                           [0,23,30],
                           [0,22,29],
                           [0,21,27],
                           [0,20,26],
                           [0,20,25],
                           [0,19,24],
                           [0,18,23],
                           [0,18,22],
                           [0,17,21],
                           [0,17,20],
                           [0,16,20],
                           [0,16,19],
                           [0,15,18],
                           [0,15,17],
                           [0,14,16],
                           [0,13,15],
                           [0,13,14],
                           [0,12,13],
                           [0,12,12],
                           [0,11,11],
                           [0,10,10],
                           [0,9,9],
                           [0,8,8],
                           [0,8,7],
                           [0,8,6],
                           [0,7,5],
                           [0,6,4],
                           [0,6,3],
                           [0,5,3],
                           [0,5,2],
                           [0,4,1],
                           [0,0,0],
                           [16,19,15],
                           [16,18,14],
                           [16,18,13],
                           [16,17,12],
                           [16,17,11],
                           [16,16,11],
                           [16,15,10],
                           [16,15,9],
                           [16,14,8],
                           [16,14,7],
                           [16,13,12],
                           [16,12,5]])
      #GLITC_delays = np.array([[0,0,0],[1,0,0]])
   
###############################################
# Do Sum correlation
###############################################

   prev_sum = 0
   this_sum = 0
   total_sum = 0
   max_total_sum = 0
   add_AB = [0] * TISC_sample_length
   add_ABC = [0] * TISC_sample_length

   best_chunk = 0
   b_lower_index_limit = 0
   c_lower_index_limit = 0
   trigger_flag = 0
   if (delay_type_flag == 1):
      square_sum_ABC = np.zeros(len(GLITC_delays))
      previous_square_sum = np.zeros(len(GLITC_delays))
   else:
      square_sum_ABC = np.zeros(200*num_samples)
      previous_square_sum = np.zeros(200*num_samples)
      delays = np.zeros((200*num_samples,3))
      i = 0

   # Use GLITC Firmware Delays
   if (delay_type_flag == 1):
      
      # Build two 16 sample sums
      for chunk in range(0,2):
         previous_square_sum = square_sum_ABC
         square_sum_ABC= np.zeros(len(GLITC_delays))

         # Get sum correlation for 16 samples
         for i in range(0,len(GLITC_delays)):
            # Determine the starting position for each sample
            a_start_pos = chunk*TISC_sample_length+GLITC_delays[i][0]
            b_start_pos = chunk*TISC_sample_length+GLITC_delays[i][1]
            c_start_pos = chunk*TISC_sample_length+GLITC_delays[i][2]
            
            # Add each sample at given delay
            add_AB = np.add(a_dig_waveform[a_start_pos:a_start_pos+TISC_sample_length],b_dig_waveform[b_start_pos:b_start_pos+TISC_sample_length])
            add_ABC = np.add(add_AB,c_dig_waveform[c_start_pos:c_start_pos+TISC_sample_length])
            square_ABC = add_ABC**2
            square_sum_ABC[i] = np.sum(square_ABC)

      # Add all the two sums for each delay
      total_sum = np.add(square_sum_ABC,previous_square_sum)
      # Find the maximum sum
      max_total_sum = np.amax(total_sum)
      # Find the delays for the largest sum
      best_delays = GLITC_delays[np.argmax(total_sum)]
      
      #print total_sum
      #print len(total_sum)
      #print len(GLITC_delays)
      #print "Max Sum: "+str(max_total_sum)
      #print best_delays

      #ab_theta = np.arcsin((speed_of_light*best_delays[1]*sample_period)/ab_distance)
      #ac_theta = np.arctan(ac_vert_distance/ac_horz_distance)-np.pi/2+np.arcsin((speed_of_light*best_delays[2]*sample_period)/ac_distance)
      #ab_theta *= 180/np.pi
      #ac_theta*= 180/np.pi

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

   if(max_total_sum > threshold):
      trigger_flag = 1
      #print "Event passes"
      
  #print "Threshold: " + str(threshold)
   #print "max_total_sum: " + str(max_total_sum)
   #print "A Chunk: " + str(best_chunk)
   #print "B Delay: " + str(best_b_delay)#+'\t'+str(b_input_delay-best_b_delay)
   #print "C Delay: " + str(best_c_delay)#+'\t'+str(c_input_delay-best_c_delay)
   #print "\n\n"
   
   return trigger_flag, max_total_sum


if __name__ == '__main__':
   import numpy as np
   from impulse import impulse_gen
   from digitizer import digitize
   import matplotlib.pyplot as plt
   threshold = 300
   num_samples = 74
   upsample = 10
   num_bits = 3
   noise_mean = 0
   noise_rms = 20
   SNR = 5
   num_upsamples = num_samples*upsample
   a_dig_waveform = np.zeros(num_upsamples)
   a_waveform = np.zeros(num_samples)
   #a_dig_waveform[20] = 3.5
   #a_dig_waveform[21] = -3.5
   #a_dig_waveform[22] = 2.5
   #a_dig_waveform[23] = -2.5
   a_delay = 20
   b_delay = 10
   c_delay = 10
   TISC_sample_length = 16
   delay_type_flag = 1
   
   signal_amp = SNR*2*noise_rms

   a_waveform = impulse_gen(num_samples,a_delay,upsample,draw_flag=0,output_dir='output/')
   
   difference=np.amax(a_waveform)-np.amin(a_waveform) # Get peak to peak voltage
   a_waveform *= (1/difference) # Normalize input
   a_waveform *= signal_amp # Amplify
   
   a_dig_waveform = digitize(a_waveform,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor=0.5)
   
   

   #print num_samples
   print a_dig_waveform
   passed_flag = sum_correlate(num_samples,a_dig_waveform,np.roll(a_dig_waveform,b_delay),np.roll(a_dig_waveform,c_delay),threshold,TISC_sample_length=TISC_sample_length,delay_type_flag=delay_type_flag)
   print passed_flag
