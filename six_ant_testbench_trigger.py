#!/usr/bin/env python
  
if __name__ == '__main__':
   debug = False
   
   import time
   import os
   import numpy as np
   import subprocess as Popen
   from array import array
   if(debug==False):
      import matplotlib
      matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   import os
   import thesis_toolbox as ttb 
   from six_antenna_sum_correlator import sum_correlate

   
   #############################################################
   # Parameters
   num_events = 10000
   samples_to_read = int(num_events*80)
   boresight = 0
   baseline = 0
   angle_range = 100
   TISC_sample_length = 16
   digitization_factor = 32.0       # Digitization factor for threshold(mV)
   noise_sigma = 28.0
   num_bits = 3                     # Number of bits available to digitizer
   simulation_rate = 162500000.0          # Simulation Rate (MHz)
   event_rate = 81250000.0                 # Rate impulsive events were generated (MHz)
   num_samples_per_window = 80                 # Length of Signal Window
   draw_flag = False                # 1=draw event graphs, 0=don't                        
   SNR_draw_flag = True             # 1=draw SNR graphs, 0=don't
   delay_type_flag = 1              # 1=use GLITC delays, 0=all possible delays
   average_subtract_flag = 1
   threshold_to_energy = 1.0#((((digitization_factor*(10**(-3)))**2)/50.0)*(1.0/2600000000.0)*32.0)*10**12 # to get to pJ
   #print threshold_to_energy
   SNR = np.array([0,1,2,3,4,5])
   trial_size = 256*32
   
   middle_delay = -15                # Ch B signal offset
   lower_delay = -17                # Ch C signal offset

   low_threshold = 0             # Lowest Threshold
   high_threshold = 4095         # Highest Threshold
   step_threshold = 1            # Threshold Interval
   
      
   impulse_amp_max = 658.42
   #array_of_impulse_atten = np.linspace(0.0,31.5,64)
   array_of_impulse_atten = np.array([100,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,50])
   if(debug==True): print array_of_impulse_atten
   array_of_SNR_values = np.zeros(len(array_of_impulse_atten))
   
   
   for i in range(len(array_of_impulse_atten)):
      array_of_SNR_values[i] = (impulse_amp_max/(2*noise_sigma))*10**(-((array_of_impulse_atten[i]*0.5))/20.0)
   if(debug==True): print array_of_SNR_values
   
   if(boresight==0):
      if(baseline==0):
         output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/testbench_analysis/output/6_ant_on_boresight_long_baseline_TISC_testbench_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
      if(baseline==1):
         output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/testbench_analysis/output/6_ant_on_boresight_short_baseline_TISC_testbench_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
   if(boresight==1):
      if(baseline==0):
         output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/testbench_analysis/output/6_ant_off_boresight_long_baseline_TISC_testbench_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
      if(baseline==1):
         output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/testbench_analysis/output/6_ant_off_boresight_short_baseline_TISC_testbench_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
         
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   #############################################################
   
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   settings_filename = open(output_dir+"/settings.txt","w")
   #print output_dir
   settings_filename.write("Input parameters for testbench trigger\n\n")
   settings_filename.write("Start time: "+str(time.strftime('%Y_%m_%d_%H.%M.%S')))
   settings_filename.write("\nNumber of samples per SNR: " + str(samples_to_read))
   settings_filename.write("\nBoresight setting: " + str(boresight))
   settings_filename.write("\nBaseline setting: " + str(baseline))
   settings_filename.write("\nAcceptable Angle Range: " + str(angle_range))
   settings_filename.write("\nNumber of samples per event: " + str(num_samples_per_window))
   settings_filename.write("\nNumber of bits: "+ str(num_bits))
   if(delay_type_flag):
      settings_filename.write("\nUsing GLITC Delays")
   else:
      settings_filename.write("\nUsing all possible event delays")
   settings_filename.write("\nCh B Input Delay: " + str(middle_delay))
   settings_filename.write("\nCh C Input Delay: " + str(lower_delay))
   settings_filename.write("\nNoise Sigma: "+str(noise_sigma))
   settings_filename.write("\nMinimum SNR: " + str(np.amin(SNR)))
   settings_filename.write("\nMaximum SNR: " + str(np.amax(SNR)))
   settings_filename.write("\nSNR Step: " + str(SNR[1]-SNR[0]))
   settings_filename.write("\nMinimum Threshold: " + str(low_threshold))
   settings_filename.write("\nMaximum Threshold: " + str(high_threshold))
   settings_filename.write("\nThreshold Step: " + str(step_threshold))
   #settings_filename.write("\nMinimum Event Rate: " + str(low_event_rate))
   #settings_filename.write("\nMaximum Event Rate: " + str(high_event_rate))
   #settings_filename.write("\nNumber of Event Rates Simulated: " + str(num_event_rate))
   settings_filename.write("\nDigitization Threshold: " + str(digitization_factor)+"*Noise_RMS")
   
   
   
  # Set some variables
   
   line_color = 1
   
   threshold = np.linspace(low_threshold,high_threshold,
                           (high_threshold-low_threshold)
                           /step_threshold+1) # Generate array of thresholds
   
   num_passed_events = np.zeros(len(threshold))
   trigger_rate = np.zeros(len(threshold))
   
   num_passed_events_uncor = np.zeros(len(threshold))
   trigger_rate_uncor = np.zeros(len(threshold))

   # Phi sectors have alternating baselines
   if(boresight==0):
      abc_impulse_amp = 1.000
      def_impulse_amp = 0.835
      #ghi_impulse_amp = 0.776
   elif(boresight==1):
      abc_impulse_amp = 0.962
      def_impulse_amp = 0.885
      #ghi_impulse_amp = 0.650
   if(baseline==0):
      abc_baseline = 0
      def_baseline = 1
      #ghi_baseline = 1
   elif(baseline==1):
      abc_baseline = 1
      def_baseline = 0
      #ghi_baseline = 0

   #num_delays = [63,46]
   #abc_correlation_mean = np.zeros(num_delays[abc_baseline])
   #def_correlation_mean = np.zeros(num_delays[def_baseline])
   #ghi_correlation_mean = np.zeros(num_delays[ghi_baseline])
   #abc_correlation_mean_uncor = np.zeros(num_delays[abc_baseline])
   #def_correlation_mean_uncor = np.zeros(num_delays[def_baseline])
   #ghi_correlation_mean_uncor = np.zeros(num_delays[ghi_baseline])
   
   SNR_counter = 0
   threshold_counter = 0
   
   ############ Run trials to get noise average #######################
   
   # Read in thermal data for avg. subtraction
   #if(debug==True): print "Reading thermal noise data"
   #time,a_uncor_therm,b_uncor_therm,c_uncor_therm,d_uncor_therm,e_uncor_therm,f_uncor_therm,a_therm,b_therm,c_therm,d_therm,e_therm,f_therm = ttb.read_data_from_data_file(100.00,samples_to_read,0)
   #if(debug==True): print "Finished reading first thermal noise data"
   #time,d_uncor_therm,e_uncor_therm,f_uncor_therm,g_uncor_therm,h_uncor_therm,i_uncor_therm,d_therm,e_therm,f_therm,g_therm,h_therm,i_therm = ttb.read_data_from_data_file(100.00,samples_to_read,1)
   #if(debug==True): print "Finished reading second thermal noise data"
   #num_thermal_samples = len(a_therm)
   """
   num_trials = 100#int(num_thermal_samples/TISC_sample_length)
   for timestep in range(1,num_trials+1):
      if(debug==True and timestep % 100==0):
         print "\nStarting trial: "+str(timestep)
      start = timestep*num_samples_per_window
      end = start+num_samples_per_window
      if(end>=num_thermal_samples):
         break
      d1,d2,d3,abc_correlation_mean,d4,d5,d6,d7 = sum_correlate(num_samples_per_window,a_therm[start:end],b_therm[start:end],c_therm[start:end],d_therm[start:end],e_therm[start:end],f_therm[start:end],abc_baseline,100,TISC_sample_length=16)
      #d1,d2,d3,def_correlation_mean,d4,d5,d6,d7 = sum_correlate(num_samples_per_window,d_therm[start:end],e_therm[start:end],f_therm[start:end],100,def_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=def_correlation_mean,trial_run_number=timestep)
      #d1,d2,d3,ghi_correlation_mean,d4,d5,d6,d7 = sum_correlate(num_samples_per_window,g_therm[start:end],h_therm[start:end],i_therm[start:end],100,ghi_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=ghi_correlation_mean,trial_run_number=timestep)
	
      d1,d2,d3,abc_correlation_mean_uncor,d4,d5,d6,d7 = sum_correlate(num_samples_per_window,a_uncor_therm[start:end],b_uncor_therm[start:end],c_uncor_therm[start:end],d_uncor_therm[start:end],e_uncor_therm[start:end],f_uncor_therm[start:end],abc_baseline,100,TISC_sample_length=16)
      #d1,d2,d3,def_correlation_mean_uncor,d4,d5,d6,d7 = sum_correlate(num_samples_per_window,d_uncor_therm[start:end],e_uncor_therm[start:end],f_uncor_therm[start:end],100,def_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=def_correlation_mean_uncor,trial_run_number=timestep)
      #d1,d2,d3,ghi_correlation_mean_uncor,d4,d5,d6,d7 = sum_correlate(num_samples_per_window,g_uncor_therm[start:end],h_uncor_therm[start:end],i_uncor_therm[start:end],100,ghi_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=ghi_correlation_mean_uncor,trial_run_number=timestep)
	
   # Set offsets for plots				
   max_offset = 100
   #print abc_correlation_mean
   ####################################################################
	"""

   # Setup output files
   data_filename = open(output_dir+"/data_output.dat","w")
   data_filename.write("time_step,SNR,abc_max,abc_as_max,abc_correlation_mean,def_max,def_as_max,def_correlation_mean,ghi_max,ghi_as_max,ghi_correlation_mean")
   graph_filename = open(output_dir+"/graph_output.dat","w")
   uncor_data_filename = open(output_dir+"/uncor_data_output.dat","w")
   uncor_data_filename.write("time_step,SNR,abc_max,abc_as_max,abc_correlation_mean,def_max,def_as_max,def_correlation_mean,ghi_max,ghi_as_max,ghi_correlation_mean")
   uncor_graph_filename = open(output_dir+"/uncor_graph_output.dat","w")
   plt.figure(1)
   plt.figure(2)
   
   #first_hist = []
   #second_hist = []
   #angle_hist = []
   #####################################################################
   for SNR_counter in range(len(SNR)):
      outside_counter=0
      #SNR_hist=[]
      #abc_hist = []
      #def_hist = []
      #ghi_hist = []
      
      
      if(debug==True): print "Starting SNR %d"%SNR[SNR_counter]
      #if(boresight==0):
      # Start working on data with impulses
      center_SNR_index,center_SNR_value = ttb.find_nearest(array_of_SNR_values,SNR[SNR_counter])
      if(SNR[SNR_counter]==0): center_SNR_value=0.00
      if(debug==True): print "Center SNR: %1.2f"%center_SNR_value
      abc_SNR_index,abc_SNR_value = ttb.find_nearest(array_of_SNR_values,SNR[SNR_counter]*abc_impulse_amp)
      abc_atten_value = array_of_impulse_atten[abc_SNR_index]
      if(debug==True): print "ABC SNR: %1.2f"%abc_SNR_value
      def_SNR_index,def_SNR_value = ttb.find_nearest(array_of_SNR_values,SNR[SNR_counter]*def_impulse_amp)
      def_atten_value = array_of_impulse_atten[def_SNR_index]
      if(debug==True): print "DEF SNR: %1.2f"%def_SNR_value
      #ghi_SNR_index,ghi_SNR_value = ttb.find_nearest(array_of_SNR_values,SNR[SNR_counter]*ghi_impulse_amp)
      #ghi_atten_value = array_of_impulse_atten[ghi_SNR_index]
      #if(debug==True): print "GHI SNR: %1.2f"%ghi_SNR_value
      if(SNR[SNR_counter]!=0):
         time,a_uncor,b_uncor,c_uncor,d_uncor,e_uncor,f_uncor,a,b,c,d,e,f = ttb.read_data_from_data_file(abc_atten_value,samples_to_read,0)
         #time,d_uncor,e_uncor,f_uncor,d1,d2,d3,d,e,f,d4,d5,d6 = ttb.read_data_from_data_file(def_atten_value,samples_to_read,1)
         #time,g_uncor,h_uncor,i_uncor,d1,d2,d3,g,h,i,d4,d5,d6 = ttb.read_data_from_data_file(ghi_atten_value,samples_to_read,2)
         """
         elif(boresight==1):
            # Start working on data with impulses
            center_SNR_index,center_SNR_value = ttb.find_nearest(array_of_SNR_values,SNR[SNR_counter])
            center_atten_value = array_of_impulse_atten[center_SNR_index]
            if(debug==True): print "Center SNR: %1.2f"%center_SNR_value
            side_SNR_index,side_SNR_value = ttb.find_nearest(array_of_SNR_values,center_SNR_value*0.9235)
            side_atten_value = array_of_impulse_atten[side_SNR_index]
            if(debug==True): print "Side SNR: %1.2f"%side_SNR_value
            far_side_SNR_index,far_side_SNR_value = ttb.find_nearest(array_of_SNR_values,center_SNR_value*0.6500)
            far_side_atten_value = array_of_impulse_atten[far_side_SNR_index]
            if(debug==True): print "Far Side SNR: %1.2f"%far_side_SNR_value
            # Read in impulsive data
            time,a_uncor,b_uncor,c_uncor,d_uncor,e_uncor,f_uncor,a,b,c,d,e,f = ttb.read_data_from_data_file(side_atten_value,samples_to_read)
            time,d_uncor,e_uncor,f_uncor,g_uncor,h_uncor,i_uncor,d,e,f,g,h,i = ttb.read_data_from_data_file(side_atten_value,samples_to_read,1)
            time,g_uncor,h_uncor,i_uncor,d1,d2,d3,g,h,i,d4,d5,d6 = ttb.read_data_from_data_file(far_side_atten_value,samples_to_read,2)
         """
         if(debug): print len(time),len(a),len(b),len(c),len(d),len(e),len(f),len(g),len(h),len(i)
         # Now shift the middle and lower rings to the correct delays
         b,c = ttb.delay_antennas(b,c,middle_delay,lower_delay)
         e,f = ttb.delay_antennas(e,f,middle_delay,lower_delay)
         #h,i = ttb.delay_antennas(h,i,middle_delay,lower_delay)
         
         b_uncor,c_uncor = ttb.delay_antennas(b_uncor,c_uncor,middle_delay,lower_delay)
         e_uncor,f_uncor = ttb.delay_antennas(e_uncor,f_uncor,middle_delay,lower_delay)
         #h_uncor,i_uncor = ttb.delay_antennas(h_uncor,i_uncor,middle_delay,lower_delay)
      
      if(SNR[SNR_counter]==0):
         thermal_noise_factor = 3
      else:
         thermal_noise_factor = 1
         
      time,a_uncor_therm,b_uncor_therm,c_uncor_therm,d_uncor_therm,e_uncor_therm,f_uncor_therm,a_therm,b_therm,c_therm,d_therm,e_therm,f_therm = ttb.read_data_from_data_file(100.00,samples_to_read*thermal_noise_factor,0)
      #print len(time)
      #time,d1,d2,d3,d_therm,e_therm,f_therm,d4,d5,d6 = ttb.read_data_from_data_file(100.00,samples_to_read*thermal_noise_factor,1)
      #print len(time)
      #time,g_uncor_therm,h_uncor_therm,i_uncor_therm,d1,d2,d3,g_therm,h_therm,i_therm,d4,d5,d6 = ttb.read_data_from_data_file(100.00,samples_to_read*thermal_noise_factor,3)
      #print len(time)

      
      
      num_samples = len(time)

      
      
      # Reset values
      trigger_rate[threshold_counter] = 0 
      time_since_last_trigger = 2.0*(1.0/event_rate)
      num_passed_events = np.zeros(len(threshold))
      #num_passed_as_events = np.zeros(len(threshold))
      
      trigger_rate_uncor[threshold_counter] = 0 
      time_since_last_trigger_uncor = 2.0*(1.0/event_rate)
      num_passed_events_uncor = np.zeros(len(threshold))
      #num_passed_as_events_uncor = np.zeros(len(threshold))
      

      # Start loop over events
      num_runs  = int(num_samples/num_samples_per_window)
      #num_events = int(num_runs/float((simulation_rate/event_rate)))
      #print "Num runs %d"%num_runs
      if(debug==True): print "Number of runs %d"%num_runs
      for timestep in range(0,num_runs):
         event_passed_flag = 0 # Reset value
         max_sum = 0
         
         start = timestep*num_samples_per_window
         end = start+num_samples_per_window
         # Don't run off the end
         if(end>=num_samples):
            break
         
         if ((timestep % int((simulation_rate/event_rate)) == 0 or SNR[SNR_counter]==0)):
            #print "thermal event"
            #print len(a_therm[start:end])
            #print len(b_therm[start:end])
            #print len(c_therm[start:end])
            trigger_flag, max_total_sum,best_delays,total_sum = sum_correlate(num_samples_per_window,a_therm[start:end],b_therm[start:end],c_therm[start:end],d_therm[start:end],e_therm[start:end],f_therm[start:end],abc_baseline,threshold,TISC_sample_length=16)
            #d1,def_max_sum,as_def_max_sum,def_correlation_mean,d4,d5,as_def_angle,def_angle = sum_correlate(num_samples_per_window,d_therm[start:end],e_therm[start:end],f_therm[start:end],threshold,def_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=def_correlation_mean)
            #d1,ghi_max_sum,as_ghi_max_sum,ghi_correlation_mean,d4,d5,as_ghi_angle,ghi_angle = sum_correlate(num_samples_per_window,g_therm[start:end],h_therm[start:end],i_therm[start:end],threshold,ghi_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=ghi_correlation_mean)
            #print as_abc_max_sum
            #print as_def_max_sum
            #print as_ghi_max_sum
            #print len(a[start:end])
            #plt.figure(8)
            #plt.plot(time[start:end],a_therm[start:end],time[start:end],d_therm[start:end],time[start:end],g_therm[start:end])
            #plt.title("Noise")
            #plt.show()
            #abc_hist.append(abc_max_sum)
            #def_hist.append(def_max_sum)
            #ghi_hist.append(ghi_max_sum)
            #SNR_hist.append(abc_max_sum+def_max_sum+ghi_max_sum)
            
            trigger_flag, max_total_sum_uncor,best_delays_uncor,total_sum_uncor = sum_correlate(num_samples_per_window,a_uncor_therm[start:end],b_uncor_therm[start:end],c_uncor_therm[start:end],d_uncor_therm[start:end],e_uncor_therm[start:end],f_uncor_therm[start:end],abc_baseline,threshold,TISC_sample_length=16)
            #d1,def_max_sum_uncor,as_def_max_sum_uncor,def_correlation_mean_uncor,d4,d5,as_def_angle_uncor,def_angle_uncor = sum_correlate(num_samples_per_window,d_uncor_therm[start:end],e_uncor_therm[start:end],f_uncor_therm[start:end],threshold,def_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=def_correlation_mean_uncor)
            #d1,ghi_max_sum_uncor,as_ghi_max_sum_uncor,ghi_correlation_mean_uncor,d4,d5,as_ghi_angle_uncor,ghi_angle_uncor = sum_correlate(num_samples_per_window,g_uncor_therm[start:end],h_uncor_therm[start:end],i_uncor_therm[start:end],threshold,ghi_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=ghi_correlation_mean_uncor)
            
         else:
            #print "Starting events runs"
            #plt.plot(time[start:end],a[start:end],time[start:end],b[start:end],time[start:end],c[start:end])
            #plt.show()
            trigger_flag, max_total_sum,best_delays,total_sum = sum_correlate(num_samples_per_window,a[start:end],b[start:end],c[start:end],d[start:end],e[start:end],f[start:end],abc_baseline,threshold,TISC_sample_length=16)
            #d1,def_max_sum,as_def_max_sum,def_correlation_mean,d4,d5,as_def_angle,def_angle = sum_correlate(num_samples_per_window,d[start:end],e[start:end],f[start:end],threshold,def_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=def_correlation_mean)
            #d1,ghi_max_sum,as_ghi_max_sum,ghi_correlation_mean,d4,d5,as_ghi_angle,ghi_angle = sum_correlate(num_samples_per_window,g[start:end],h[start:end],i[start:end],threshold,ghi_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=ghi_correlation_mean)
            #print as_abc_max_sum
            #print end-start
            #print len(a[start:end])
            #if(debug):
               #print abc_angle,abc_max_sum
               #print def_angle,def_max_sum
               #print ghi_angle,ghi_max_sum
            #angle_hist.append(abc_angle)
            #angle_hist.append(def_angle)
            #angle_hist.append(ghi_angle)
            """
            plt.figure(9)
            plt.plot(time[start:end],a[start:end],time[start:end],b[start:end],time[start:end],c[start:end])
            plt.title("ABC")
            plt.figure(10)
            plt.plot(time[start:end],d[start:end],time[start:end],e[start:end],time[start:end],f[start:end])
            plt.title("DEF")
            plt.figure(11)
            plt.plot(time[start:end],g[start:end],time[start:end],h[start:end],time[start:end],i[start:end])
            plt.title("GHI")
            plt.show()
            """
            #abc_hist.append(abc_max_sum)
            #def_hist.append(def_max_sum)
            #ghi_hist.append(ghi_max_sum)
            trigger_flag, max_total_sum_uncor,best_delays_uncor,total_sum_uncor = sum_correlate(num_samples_per_window,a_uncor[start:end],b_uncor[start:end],c_uncor[start:end],d_uncor[start:end],e_uncor[start:end],f_uncor[start:end],abc_baseline,threshold,TISC_sample_length=16)
            #d1,def_max_sum_uncor,as_def_max_sum_uncor,def_correlation_mean_uncor,d4,d5,as_def_angle_uncor,def_angle_uncor = sum_correlate(num_samples_per_window,d_uncor[start:end],e_uncor[start:end],f_uncor[start:end],threshold,def_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=def_correlation_mean_uncor)
            #d1,ghi_max_sum_uncor,as_ghi_max_sum_uncor,ghi_correlation_mean_uncor,d4,d5,as_ghi_angle_uncor,ghi_angle_uncor = sum_correlate(num_samples_per_window,g_uncor[start:end],h_uncor[start:end],i_uncor[start:end],threshold,ghi_baseline,TISC_sample_length=16,delay_type_flag=1,average_subtract_flag=1,correlation_mean=ghi_correlation_mean_uncor)
          
          
          
         """   
         #as_total_correlated_sum = as_abc_max_sum+as_def_max_sum+as_ghi_max_sum
         #total_correlated_sum = abc_max_sum+def_max_sum+ghi_max_sum
         #as_total_correlated_sum_uncor = as_abc_max_sum_uncor+as_def_max_sum_uncor+as_ghi_max_sum_uncor
         #total_correlated_sum_uncor = abc_max_sum_uncor+def_max_sum_uncor+ghi_max_sum_uncor
         #print as_total_correlated_sum
         
         #first_angle_range = np.abs(np.amax([as_abc_angle,as_def_angle])-np.min([as_abc_angle,as_def_angle]))
         #print first_angle_range
         #first_hist.append(first_angle_range)
         #second_angle_range = np.abs(np.amax([as_abc_angle,as_ghi_angle])-np.min([as_abc_angle,as_ghi_angle]))
         #print second_angle_range
         #first_angle_range_uncor = np.abs(np.amax([as_abc_angle_uncor,as_def_angle_uncor])-np.min([as_abc_angle_uncor,as_def_angle_uncor]))
         #first_hist.append(first_angle_range)
         #second_angle_range_uncor = np.abs(np.amax([as_abc_angle_uncor,as_ghi_angle_uncor])-np.min([as_abc_angle_uncor,as_ghi_angle_uncor]))
         #second_hist.append(second_angle_range)
         
         
         #if(debug):
            #print "First Angle range: %1.2f"%(first_angle_range)
            #print "Second Angle range: %1.2f"%(second_angle_range)
            #print abc_angle
            #print def_angle
            #print ghi_angle
            #print "\n"
         
         
         
         if((first_angle_range>angle_range) and (second_angle_range>angle_range)):
            outside_counter+=1
            if(debug): print first_angle_range
            if(debug): print second_angle_range
            as_total_correlated_sum=0
            total_correlated_sum=0
         if((first_angle_range_uncor>angle_range)and(second_angle_range_uncor>angle_range)):
            as_total_correlated_sum_uncor=0
            total_correlated_sum_uncor=0
         """
         data_filename.write('\n'+str(center_SNR_value)+','+str(max_total_sum))
         #uncor_data_filename.write('\n'+str(center_SNR_value)+','+str(abc_max_sum_uncor)+','+str(as_abc_max_sum_uncor)+','+str(def_max_sum_uncor)+','+str(as_def_max_sum_uncor)+','+str(ghi_max_sum_uncor)+','+str(as_ghi_max_sum_uncor))
         uncor_data_filename.write('\n'+str(center_SNR_value)+','+str(max_total_sum_uncor))   
         #print three_phi_max_sum
         for threshold_counter in range(0,len(threshold)):

            if((max_total_sum)>threshold[threshold_counter]):
               num_passed_events[threshold_counter] += 1
               #print num_passed_events[threshold_counter]
            if((max_total_sum_uncor)>threshold[threshold_counter]):
               num_passed_events_uncor[threshold_counter] += 1
               #print num_passed_events[threshold_counter]
      

      """
      #plt.figure(3)
      #plt.clf()
      #plt.hist(abc_hist)
      
      
      plt.title("ABC Power at SNR= %d (N=%d, mu=%.2f,sigma=%.2f)"%(SNR[SNR_counter],num_events,np.mean(abc_hist),np.sqrt(np.var(abc_hist))))
      plt.savefig(str(output_dir)+"/abc_SNR_%.2f_histogram.png"%SNR[SNR_counter])
      plt.figure(4)
      plt.clf()
      plt.hist(def_hist)
      plt.title("DEF Power at SNR= %d (N=%d, mu=%.2f,sigma=%.2f)"%(SNR[SNR_counter],num_events,np.mean(def_hist),np.sqrt(np.var(def_hist))))
      plt.savefig(str(output_dir)+"/def_SNR_%.2f_histogram.png"%SNR[SNR_counter])
      plt.figure(5)
      plt.clf()
      plt.hist(ghi_hist)
      plt.title("GHI Power at SNR= %d (N=%d, mu=%.2f,sigma=%.2f)"%(SNR[SNR_counter],num_events,np.mean(ghi_hist),np.sqrt(np.var(ghi_hist))))
      plt.savefig(str(output_dir)+"/ghi_SNR_%.2f_histogram.png"%SNR[SNR_counter])
      #plt.show()
      #if(debug):
         #print len(abc_hist)
         #print len(def_hist)
         #print len(ghi_hist)
         #print outside_counter
      #plt.figure(3)
      #plt.hist(SNR_hist,bins=50)
      #plt.title("SNR %d (N=%d, mu=%.2f,sigma=%.2f"%(SNR[SNR_counter],num_events,np.mean(SNR_hist),np.sqrt(np.var(SNR_hist))))
      #plt.show(3)
      #plt.hist(first_hist)
      #plt.hist(second_hist)
      #plt.show(3)
      """
      # Calculate trigger rate for current threshold
      
      for threshold_counter in range(0,len(threshold)):
         trigger_rate[threshold_counter] = (float(num_passed_events[threshold_counter])*float(simulation_rate))/float(num_runs)
         graph_filename.write(str(center_SNR_value)+','+str(threshold[threshold_counter])+','+str(trigger_rate[threshold_counter])+'\n')
         trigger_rate_uncor[threshold_counter] = (float(num_passed_events_uncor[threshold_counter])*float(simulation_rate))/float(num_runs)
         uncor_graph_filename.write(str(center_SNR_value)+','+str(threshold[threshold_counter])+','+str(trigger_rate_uncor[threshold_counter])+'\n')
         #print str(SNR[SNR_counter])+'\t'+str(threshold[threshold_counter])+'\t'+str(trigger_rate[threshold_counter])

     

      # Close Threshold Loop 

      # Make and save corrected SNR curve
      plt.figure(1)
      plt.semilogy(threshold*threshold_to_energy,trigger_rate,label=('SNR %1.2f '%center_SNR_value))
      plt.figure(2)
      plt.semilogy(threshold*threshold_to_energy,trigger_rate_uncor,label=('SNR %1.2f '%center_SNR_value))
      #if(SNR[SNR_counter]==0.0):
         #print threshold
        #print trigger_rate
         #fit_start,a,b = ttb.fit_exponential_tail(threshold,trigger_rate,debug=True)
         #print fit
         #print threshold[SNR_start+fit_start:SNR_end]
         #plt.semilogy(threshold[fit_start:],np.exp(a*np.array(threshold[fit_start:])+b),linestyle=":",color='k')
   plt.figure(1)
   plt.legend()
   plt.grid(True)
   plt.xlabel("Threshold [DAC]")
   plt.ylabel("Trigger Rate [Hz]")
   plt.figure(2)
   plt.legend()
   plt.grid(True)
   plt.xlabel("Threshold [DAC]")
   plt.ylabel("Trigger Rate [Hz]")
   #if(cw_flag):
      #plt.title("TISC SNR Curves, Simulation Rate: %1.2fMHz, Event Rate: %1.2fMHz, CW Amp: %d*Noise RMS" % (simulation_rate,event_rate,cw_amplitude/noise_sigma))
   #else:
   if(boresight==0):
      if(baseline==0):
         plt.figure(1)
         plt.title("6 Antenna On Phi Sector, System Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate/10**6,event_rate/10**6))
         plt.figure(2)
         plt.title("Uncorrected 6 Antenna On Phi Sector, System Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate/10**6,event_rate/10**6))
      elif(baseline==1):
         plt.figure(1)
         plt.title("6 Antenna On Phi Sector, System Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate/10**6,event_rate/10**6))
         plt.figure(1)
         plt.title("Uncorrected 6 Antenna On Phi Sector, System Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate/10**6,event_rate/10**6))
   if(boresight==1):
      if(baseline==0):
         plt.figure(1)
         plt.title("6 Antenna Between Phi Sector, Symvstem Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate/10**6,event_rate/10**6))
         plt.figure(2)
         plt.title("6 Antenna Between Phi Sector, System Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate/10**6,event_rate/10**6))
      elif(baseline==1):
         plt.figure(1)
         plt.title("6 Antenna Between Phi Sector, System Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate/10**6,event_rate/10**6))
         plt.figure(2)
         plt.title("6 Antenna Between Phi Sector, System Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate/10**6,event_rate/10**6))
   plt.figure(1)
   plt.axis([low_threshold*threshold_to_energy,high_threshold*threshold_to_energy,1.0,200000000.0])
   plt.figure(2)
   plt.axis([low_threshold*threshold_to_energy,high_threshold*threshold_to_energy,1.0,200000000.0])
   plt.figure(1)
   plt.savefig(str(output_dir)+"/TISC_SNR_canvas.png")
   plt.figure(2)
   plt.savefig(str(output_dir)+"/uncor_TISC_SNR_canvas.png")
   data_filename.close()
   """
   plt.figure(10)
   #print len(angle_hist)
   plt.hist(angle_hist,bins=25)
   plt.axvline(-10.0)
   plt.title("Impulsive Event Angle Reconstruction")
   plt.xlabel("Elevation Angle [degrees]")
   plt.ylabel("Counts")
   plt.savefig(str(output_dir)+"/angle_histogram.png")
   """
   if(debug==True): plt.show()

   # Save files and close
