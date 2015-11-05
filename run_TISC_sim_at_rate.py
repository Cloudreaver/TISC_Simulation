#!/usr/bin/env python

if __name__ == '__main__':
   import time
   import os
   from TISC_sim import TISC_sim
   import numpy as np
   import subprocess as Popen
   from array import array
   
   #############################################################
   # Parameters
   num_events = 2            # Number of events to generate per loop
   upsample = 10                     # Upsamping factor
   digitization_factor = 32.0       # Digitization factor for threshold(mV)
   num_bits = 3                     # Number of bits available to digitizer
   simulation_rate = 162500000.0          # Simulation Rate
   event_rate=81250000.0          # Rate to generate events
   num_runs = int(num_events*(simulation_rate/event_rate)) 
   num_samples = 80                 # Length of Signal Window
   noise_sigma = 32.0               # Noise Sigma in mV
   draw_flag = False                # 1=draw event graphs, 0=don't                        
   SNR_draw_flag = True             # 1=draw SNR graphs, 0=don't
   delay_type_flag = 1              # 1=use GLITC delays, 0=all possible delays
   cw_flag = False
   carrier_frequency = 260000000.0    # Hz
   modulation_frequency = 1000.0  # Hz
   cw_rms = 2.0*noise_sigma #Peak amplitude in mV
   average_subtract_flag = 1
   num_trials = 100
   threshold_to_energy = ((((digitization_factor*(10**(-3)))**2)/50.0)*(1.0/2600000000.0)*32.0)*10**12 # to get to pJ
   

   # These delays should be negative, since A is the top antenna
   # and we expect an upgoing signal
   b_input_delay = 0                # Ch B signal offset
   c_input_delay = 0                # Ch C signal offset

   low_SNR = 0.0                    # Lowest SNR
   high_SNR = 5.0                   # Highest SNR
   step_SNR = 0.5                 # SNR interval

   low_threshold = 0              # Lowest Threshold
   high_threshold = 1000          # Highest Threshold
   step_threshold = 1              # Threshold Interval
   #############################################################
   
   # Set some variables
   num_upsamples = upsample*num_samples # Number of upsampled samples
   line_color = 1
   
   SNR_Hist = {} # Empty array
   SNR_Graph = {} # Empty array
   threshold = np.linspace(low_threshold,high_threshold,
                           (high_threshold-low_threshold)
                           /step_threshold+1) # Generate array of thresholds
   
   num_passed_events = np.zeros(len(threshold))
   trigger_rate = np.zeros(len(threshold))


   SNR = np.linspace(low_SNR,high_SNR,((high_SNR-low_SNR)/step_SNR)+1)
   
   correlation_mean = np.zeros(44)

   # Write settings file
   if(cw_flag):
      output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/1_phi_sector_TISC_SIM_CW"+str(cw_rms)+"_Num"+str(num_runs)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
   else:
      output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/1_phi_sector_TISC_SIM_"+"Num"+str(num_runs)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   settings_filename = open(output_dir+"/settings.txt","w")
   #print output_dir
   settings_filename.write("Input parameters for TISC simulation\n\n")
   settings_filename.write("Start time: "+str(time.strftime('%Y_%m_%d_%H.%M.%S')))
   settings_filename.write("\nNumber of events per run: " + str(num_runs))
   settings_filename.write("\nNumber of samples per event: " + str(num_samples))
   settings_filename.write("\nNumber of bits: "+ str(num_bits))
   if(delay_type_flag):
      settings_filename.write("\nUsing GLITC Delays")
   else:
      settings_filename.write("\nUsing all possible event delays")
   if(cw_flag):
      settings_filename.write("\nAdding CW Noise")
      settings_filename.write("\nCW Frequency: " +str(carrier_frequency))
      settings_filename.write("\nModulation Frequency: " + str(modulation_frequency))
      settings_filename.write("\nCW Amplitude: " + str(cw_rms))
   
   settings_filename.write("\nCh B Input Delay: " + str(b_input_delay))
   settings_filename.write("\nCh C Input Delay: " + str(c_input_delay))
   settings_filename.write("\nNoise Sigma: "+str(noise_sigma))
   settings_filename.write("\nMinimum SNR: " + str(low_SNR))
   settings_filename.write("\nMaximum SNR: " + str(high_SNR))
   settings_filename.write("\nSNR Step: " + str(step_SNR))
   settings_filename.write("\nMinimum Threshold: " + str(low_threshold))
   settings_filename.write("\nMaximum Threshold: " + str(high_threshold))
   settings_filename.write("\nThreshold Step: " + str(step_threshold))
   #settings_filename.write("\nMinimum Event Rate: " + str(low_event_rate))
   #settings_filename.write("\nMaximum Event Rate: " + str(high_event_rate))
   #settings_filename.write("\nNumber of Event Rates Simulated: " + str(num_event_rate))
   settings_filename.write("\nNumber of Runs: " + str(num_runs))
   settings_filename.write("\nUpsample Factor: " + str(upsample))
   settings_filename.write("\nDigitization Threshold: " + str(digitization_factor)+"*Noise_RMS")
   
   SNR_counter = 0
   threshold_counter = 0

   ############ Run trials to get noise average #######################
   for timestep in range(1,num_trials+1):
      #if(timestep % 10 == 0):
         #print "\nStarting trial: "+str(timestep)

      if(cw_flag):		
			# CW + thermal noise event
			#print "cw+thermal"		
         d1, d2, d3, correlation_mean,d1,d2 = TISC_sim(0.0,100,
					b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
					cw_flag=cw_flag,carrier_frequency=carrier_frequency,cw_rms=cw_rms,modulation_frequency=modulation_frequency,
					noise_sigma=noise_sigma,average_subtract_flag=1,
					correlation_mean=correlation_mean,trial_run_number=timestep,digitization_factor=digitization_factor)
      else:
         # Thermal event					
         d1, d2, d3, correlation_mean,d1,d2= TISC_sim(0.0,100,
               b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
               cw_flag=0,carrier_frequency=carrier_frequency,cw_rms=cw_rms,modulation_frequency=modulation_frequency,
               noise_sigma=noise_sigma,average_subtract_flag=1,correlation_mean=correlation_mean,
               trial_run_number=timestep,digitization_factor=digitization_factor)
			#print "#########################"			
	# Set offsets for plots				
   max_offset = 100

	####################### Step over time #############################

   data_filename = open(output_dir+"/graph_output.dat","w")
   
   #threshold_for_branch = np.zeros(1)
   # Start Loop over SNR values
   for SNR_counter in range(0,len(SNR)):

      
      # Start loop over threshold values
      
         # Reset values
      trigger_rate[threshold_counter] = 0 
      time_since_last_trigger = 2.0*(1.0/event_rate)
      num_passed_events = np.zeros(len(threshold))
      

      # Start loop over events
      for timestep in range(0,num_runs):
         event_passed_flag = 0 # Reset value
         max_sum = 0
         #print "Starting events runs"
         # Check to see if trigger is available
         if ((timestep % int((simulation_rate/event_rate)) == 0)):
            #print "Generating impulsive event"
            event_passed_flag, d1, max_sum, correlation_mean, d2,d3 = TISC_sim(SNR[SNR_counter],100,
						b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,cw_rms=cw_rms,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,average_subtract_flag=1,
						correlation_mean=correlation_mean,trial_run_number=0,digitization_factor=digitization_factor)
         else:
            #print "Generating Noise Event"
            event_passed_flag,d1, max_sum,correlation_mean,d2,d3 = TISC_sim(0.0,100,
						b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,cw_rms=cw_rms,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,average_subtract_flag=1,
						correlation_mean=correlation_mean,trial_run_number=0,digitization_factor=digitization_factor)
      
         for threshold_counter in range(0,len(threshold)):
            if((max_sum+max_offset)>threshold[threshold_counter]):
               #threshold_for_branch[0]=threshold[threshold_counter]
               num_passed_events[threshold_counter] += 1
               #print num_passed_events[threshold_counter]


      # Calculate trigger rate for current threshold
      for threshold_counter in range(0,len(threshold)):
         trigger_rate[threshold_counter] = (float(num_passed_events[threshold_counter])/float(num_runs))*simulation_rate
         data_filename.write(str(SNR[SNR_counter])+','+str(threshold[threshold_counter])+','+str(trigger_rate[threshold_counter])+'\n')
         #print str(SNR[SNR_counter])+'\t'+str(threshold[threshold_counter])+'\t'+str(trigger_rate[threshold_counter])

     

      # Close Threshold Loop 
      
   # Close SNR Loop

   # Save files and close
   #outfile.Write()
   #outfile.Close()
   #outfile.ls()
   settings_filename.write("\nEnd time: "+str(time.strftime('%Y_%m_%d_%H.%m.%S')))
   settings_filename.close()
   data_filename.close()
   #if(SNR_draw_flag):
      #dummy = raw_input('Press any key to close')
