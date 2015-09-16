#!/usr/bin/env python

if __name__ == '__main__':
   #from ROOT import TGraph, TMultiGraph,TCanvas,TImage,gROOT,gStyle,gSystem,gPad,TFile,TLegend, TH1F, TTree
   import time
   import os
   from two_phi_sector_TISC_sim import TISC_sim
   #from impulse import impulse_gen
   import numpy as np
   import subprocess as Popen
   from array import array
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   
   #############################################################
   # Parameters
   num_events = 20000       # Number of events to generate per loop
   upsample = 10                     # Upsamping factor
   digitization_factor = 20.0       # Digitization factor for threshold(mV)
   num_bits = 3                     # Number of bits available to digitizer
   simulation_rate = 200.0          # Simulation Rate
   event_rate=100.0                 # Rate to generate events
   num_runs = int(num_events*(simulation_rate/event_rate)) 
   num_samples = 80                 # Length of Signal Window
   noise_sigma = 20.0               # Noise Sigma in mV
   draw_flag = False                # 1=draw event graphs, 0=don't                        
   SNR_draw_flag = True             # 1=draw SNR graphs, 0=don't
   delay_type_flag = 1              # 1=use GLITC delays, 0=all possible delays
   cw_flag = True
   carrier_frequency = 260000000.0    # Hz
   modulation_frequency = 1000.0  # Hz
   cw_amplitude = 2.0*noise_sigma #Peak amplitude in mV
   average_subtract_flag = 1
   num_trials = 100
   threshold_to_energy = ((1./2.6)/50.) # to get to nJ

   # These delays should be negative, since A is the top antenna
   # and we expect an upgoing signal
   b_input_delay = 0                # Ch B signal offset
   c_input_delay = 0                # Ch C signal offset

   low_SNR = 0.0                    # Lowest SNR
   high_SNR = 5.0                   # Highest SNR
   step_SNR = 0.5                 # SNR interval

   low_threshold = 0           # Lowest Threshold
   high_threshold = 1000         # Highest Threshold
   step_threshold = 1            # Threshold Interval
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
   
   abc_correlation_mean = np.zeros(44)
   def_correlation_mean = np.zeros(44)
   ghi_correlation_mean = np.zeros(44)

   # Write settings file
   if(cw_flag):
      output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/3_phi_TISC_SIM_CW"+str(cw_amplitude)+"_Num"+str(num_runs)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
   else:
      output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/3_phi_TISC_SIM_"+"Num"+str(num_runs)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
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
      settings_filename.write("\nCW Amplitude: " + str(cw_amplitude))
   else:
      cw_amplitude=0.0
      carrier_frequency=0.0
      modulation_frequency=0.0
   
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
         d1,d2,d3,d4,d5,d6, abc_correlation_mean,def_correlation_mean,ghi_correlation_mean = TISC_sim(0.0,100,
					b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
					cw_flag=cw_flag,carrier_frequency=carrier_frequency,cw_amplitude=cw_amplitude,modulation_frequency=modulation_frequency,
					noise_sigma=noise_sigma,average_subtract_flag=1,
					abc_correlation_mean=abc_correlation_mean,def_correlation_mean=def_correlation_mean,ghi_correlation_mean=ghi_correlation_mean,
               trial_run_number=timestep,digitization_factor=digitization_factor)
      else:
         # Thermal event					
         d1,d2,d3,d4,d5,d6, abc_correlation_mean,def_correlation_mean,ghi_correlation_mean= TISC_sim(0.0,100,
               b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
               cw_flag=0,carrier_frequency=carrier_frequency,cw_amplitude=cw_amplitude,modulation_frequency=modulation_frequency,
               noise_sigma=noise_sigma,average_subtract_flag=1,
               abc_correlation_mean=abc_correlation_mean,def_correlation_mean=def_correlation_mean,ghi_correlation_mean=ghi_correlation_mean,
               trial_run_number=timestep,digitization_factor=digitization_factor)
			#print "#########################"			
	# Set offsets for plots				
   max_offset = 100
   #print abc_correlation_mean
	####################### Step over time #############################

   graph_filename = open(output_dir+"/graph_output.dat","w")
   graph_filename.write("SNR,threshold,trigger_rate\n")
   data_filename = open(output_dir+"/data_output.dat","w")
   data_filename.write("time_step,SNR,thermal_noise_sigma,cw_amplitude,carrier_frequency,modulation_frequency,abc_max,abc_as_max,abc_correlation_mean,def_max,def_as_max,def_correlation_mean,ghi_max,ghi_as_max,ghi_correlation_mean")
   plt.figure(figsize=(16,12))
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
            abc_max_sum,abc_as_max_sum,def_max_sum,def_as_max_sum,ghi_max_sum,ghi_as_max_sum, abc_correlation_mean,def_correlation_mean,ghi_correlation_mean = TISC_sim(SNR[SNR_counter],100,
						b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,cw_amplitude=cw_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,average_subtract_flag=1,
						abc_correlation_mean=abc_correlation_mean,def_correlation_mean=def_correlation_mean,ghi_correlation_mean=ghi_correlation_mean,
                  trial_run_number=0,digitization_factor=digitization_factor)
            data_filename.write('\n'+str(timestep)+','+str(0.0)+','+str(noise_sigma)+','+str(cw_amplitude)+','+str(carrier_frequency)+','+str(modulation_frequency)+','+str(abc_max_sum)+','+str(abc_as_max_sum)+','+str(def_max_sum)+','+str(def_as_max_sum)+','+str(ghi_max_sum)+','+str(ghi_as_max_sum))
            #print abc_as_max_sum
            #print def_as_max_sum
            #print ghi_as_max_sum
            #print abc_as_max_sum+def_as_max_sum+ghi_as_max_sum
         else:
            #print "Generating Noise Event"
            abc_max_sum,abc_as_max_sum,def_max_sum,def_as_max_sum,ghi_max_sum,ghi_as_max_sum, abc_correlation_mean,def_correlation_mean,ghi_correlation_mean = TISC_sim(0.0,100,
						b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,cw_amplitude=cw_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,average_subtract_flag=1,
						abc_correlation_mean=abc_correlation_mean,def_correlation_mean=def_correlation_mean,ghi_correlation_mean=ghi_correlation_mean,
                  trial_run_number=0,digitization_factor=digitization_factor)
            data_filename.write('\n'+str(timestep)+','+str(SNR[SNR_counter])+','+str(noise_sigma)+','+str(cw_amplitude)+','+str(carrier_frequency)+','+str(modulation_frequency)+','+str(abc_max_sum)+','+str(abc_as_max_sum)+','+str(def_max_sum)+','+str(def_as_max_sum)+','+str(ghi_max_sum)+','+str(ghi_as_max_sum))
            
         three_phi_max_sum = abc_as_max_sum+def_as_max_sum+ghi_as_max_sum
         #print three_phi_max_sum
         for threshold_counter in range(0,len(threshold)):
            #print '\n'
            #print threshold[threshold_counter]
            #print abc_as_max_sum
            #print def_as_max_sum
            #print ghi_as_max_sum
            #print three_phi_max_sum 
            if((three_phi_max_sum+max_offset)>threshold[threshold_counter]):
               #threshold_for_branch[0]=threshold[threshold_counter]
               num_passed_events[threshold_counter] += 1
               #print num_passed_events[threshold_counter]
 
      # Calculate trigger rate for current threshold
      for threshold_counter in range(0,len(threshold)):
         trigger_rate[threshold_counter] = (float(num_passed_events[threshold_counter])/float(num_runs))*simulation_rate
         graph_filename.write(str(SNR[SNR_counter])+','+str(threshold[threshold_counter])+','+str(trigger_rate[threshold_counter])+'\n')
         #print str(SNR[SNR_counter])+'\t'+str(threshold[threshold_counter])+'\t'+str(trigger_rate[threshold_counter])

     

      # Close Threshold Loop 
      #print threshold
      #print trigger_rate
      plt.semilogy(threshold,trigger_rate,label='SNR: '+str(SNR[SNR_counter]))
   plt.legend()
   plt.grid(True)
   plt.xlabel("Threshold [unitless]")
   plt.ylabel("Trigger Rate [Hz]")
   if(cw_flag):
      plt.title("TISC SNR Curves, Simulation Rate: %1.2fMHz, Event Rate: %1.2fMHz, CW Amp: %d*Noise RMS" % (simulation_rate,event_rate,cw_amplitude/noise_sigma))
   else:
      plt.title("TISC SNR Curves, Simulation Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (simulation_rate,event_rate))
   plt.axis([low_threshold,high_threshold,10**(-2),300])
   #plt.show()
   plt.savefig(str(output_dir)+"/TISC_SNR_canvas.png")
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
