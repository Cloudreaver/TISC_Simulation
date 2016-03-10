#!/usr/bin/env python

if __name__ == '__main__':
   import time
   import os
   from three_phi_sector_TISC_sim import TISC_sim
   import numpy as np
   import subprocess as Popen
   from array import array
   from simulation_parameters import sim_param
   if(sim_param.debug==False):    
      import matplotlib
      matplotlib.use('Agg')
   
   import matplotlib.pyplot as plt
   
   #############################################################
   # Parameters
   if(False):
      num_events = 10000   # Number of events to generate per loop
      boresight = 0
      baseline = 0
      angle_range = 100
      upsample = 10                     # Upsamping factor
      digitization_factor = 32.0       # Digitization factor for threshold(mV)
      num_bits = 3                     # Number of bits available to digitizer
      simulation_rate = 162500000.0          # Simulation Rate
      event_rate = 81250000.0                 # Rate to generate events
      
      num_samples = 80                 # Length of Signal Window
      noise_sigma = 32.0               # Noise Sigma in mV
      draw_flag = False                # 1=draw event graphs, 0=don't                        
      SNR_draw_flag = True             # 1=draw SNR graphs, 0=don't
      delay_type_flag = 1              # 1=use GLITC delays, 0=all possible delays
      cw_flag = False
      carrier_frequency = 260000000.0    # Hz
      modulation_frequency = 1000.0      # Hz
      cw_amplitude = 2.0*noise_sigma #Peak amplitude in mV
      average_subtract_flag = 0
      num_trials = 2
      threshold_to_energy = 1.0#((((digitization_factor*(10**(-3)))**2)/50.0)*(1.0/2600000000.0)*32.0)*10**12 # to get to pJ
      
      # These delays should be negative, since A is the top antenna
      # and we expect an upgoing signal
      b_input_delay = -15                # Ch B signal offset
      c_input_delay = -17           # Ch C signal offset

      low_SNR = 0.0                   # Lowest SNR
      high_SNR = 5.0                 # Highest SNR
      step_SNR = 1.0                # SNR interval

      low_threshold = 0             # Lowest Threshold
      high_threshold = 4095        # Highest Threshold
      step_threshold = 1            # Threshold Interval
      
      window_length = 8
      window_weight = 0.5
   #############################################################
   
   # Set some variables
   num_runs = int(sim_param.num_events*(sim_param.simulation_rate/sim_param.event_rate)) 
   num_upsamples = sim_param.upsample*sim_param.num_samples # Number of upsampled samples
   line_color = 1
   
   threshold = np.linspace(sim_param.low_threshold,sim_param.high_threshold,
                           (sim_param.high_threshold-sim_param.low_threshold)
                           /sim_param.step_threshold+1) # Generate array of thresholds
   num_passed_events = np.zeros(len(threshold))
   trigger_rate = np.zeros(len(threshold))
   SNR = np.linspace(sim_param.low_SNR,sim_param.high_SNR,((sim_param.high_SNR-sim_param.low_SNR)/sim_param.step_SNR)+1)
   num_delays = [63,46]
   
   # Phi sectors have alternating baselines
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
         
   # Write settings file
   if(sim_param.cw_flag):
      output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/on_boresight_TISC_SIM_CW"+str(cw_amplitude)+"_Num"+str(num_runs)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
   else:
      if(sim_param.boresight==0):
         if(sim_param.baseline==0):
            output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/on_boresight_long_baseline_TISC_SIM_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
         if(sim_param.baseline==1):
            output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/on_boresight_short_baseline_TISC_SIM_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
      if(sim_param.boresight==1):
         if(sim_param.baseline==0):
            output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/off_boresight_long_baseline_TISC_SIM_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
         if(sim_param.baseline==1):
            output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/off_boresight_short_baseline_TISC_SIM_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
            
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   settings_filename = open(output_dir+"/settings.txt","w")
   settings_filename.write("Input parameters for TISC simulation\n\n")
   settings_filename.write("Start time: "+str(time.strftime('%Y_%m_%d_%H.%M.%S')))
   settings_filename.write("\nNumber of events per run: " + str(num_runs))
   settings_filename.write("\nBoresight setting: " + str(sim_param.boresight))
   settings_filename.write("\nBaseline setting: " + str(sim_param.baseline))
   settings_filename.write("\nAcceptable Angle Range: " + str(sim_param.angle_range))
   settings_filename.write("\nNumber of samples per event: " + str(sim_param.num_samples))
   settings_filename.write("\nNumber of bits: "+ str(sim_param.num_bits))
   if(sim_param.delay_type_flag):
      settings_filename.write("\nUsing GLITC Delays")
   else:
      settings_filename.write("\nUsing all possible event delays")
   if(sim_param.cw_flag):
      settings_filename.write("\nAdding CW Noise")
      settings_filename.write("\nCW Frequency: " +str(sim_param.carrier_frequency))
      settings_filename.write("\nModulation Frequency: " + str(sim_param.modulation_frequency))
      settings_filename.write("\nCW Amplitude: " + str(sim_param.cw_amplitude))   
   settings_filename.write("\nCh B Input Delay: " + str(sim_param.b_input_delay))
   settings_filename.write("\nCh C Input Delay: " + str(sim_param.c_input_delay))
   settings_filename.write("\nNoise Sigma: "+str(sim_param.noise_sigma))
   settings_filename.write("\nMinimum SNR: " + str(sim_param.low_SNR))
   settings_filename.write("\nMaximum SNR: " + str(sim_param.high_SNR))
   settings_filename.write("\nSNR Step: " + str(sim_param.step_SNR))
   settings_filename.write("\nMinimum Threshold: " + str(sim_param.low_threshold))
   settings_filename.write("\nMaximum Threshold: " + str(sim_param.high_threshold))
   settings_filename.write("\nThreshold Step: " + str(sim_param.step_threshold))
   settings_filename.write("\nNumber of Runs: " + str(num_runs))
   settings_filename.write("\nUpsample Factor: " + str(sim_param.upsample))
   settings_filename.write("\nDigitization Threshold: " + str(sim_param.digitization_factor)+"*Noise_RMS")
   if(sim_param.window_length):
      settings_filename.write("\nWindow Length: " + str(sim_param.window_length))
      settings_filename.write("\nWindow Weight: " + str(sim_param.window_weight))
   
   SNR_counter = 0
   threshold_counter = 0

	####################### Step over time #############################

   graph_filename = open(output_dir+"/graph_output.dat","w")
   graph_filename.write("SNR,threshold,trigger_rate\n")
   data_filename = open(output_dir+"/data_output.dat","w")
   data_filename.write("time_step,SNR,thermal_noise_sigma,cw_amplitude,carrier_frequency,modulation_frequency,abc_max,abc_as_max,abc_correlation_mean,def_max,def_as_max,def_correlation_mean,ghi_max,ghi_as_max,ghi_correlation_mean")

   # Start Loop over SNR values
   for SNR_counter in range(0,len(SNR)):
      first_hist = []
      second_hist = []
      sum_hist = []
            
      # Reset values
      trigger_rate[threshold_counter] = 0 
      time_since_last_trigger = 2.0*(1.0/sim_param.event_rate)
      num_passed_events = np.zeros(len(threshold))
      
      # Start loop over events
      for timestep in range(0,num_runs):
         event_passed_flag = 0 # Reset value
         max_sum = 0
      
         # Check to see if trigger is available
         if ((timestep % int((sim_param.simulation_rate/sim_param.event_rate)) == 0)):
            abc_max_sum,def_max_sum,ghi_max_sum,abc_angle,def_angle,ghi_angle = TISC_sim(sim_param,SNR=sim_param.SNR,seed=timestep)
            data_filename.write('\n'+str(timestep)+','+str(0.0)+','+str(sim_param.noise_sigma)+','+str(sim_param.cw_amplitude)+','+str(sim_param.cw_frequency)+','+str(sim_param.modulation_frequency)+','+str(abc_max_sum)+','+str(def_max_sum)+','+str(ghi_max_sum))
         else:
            abc_max_sum,def_max_sum,ghi_max_sum,abc_angle,def_angle,ghi_angle = TISC_sim(sim_param,SNR=0.0,seed=timestep)
            data_filename.write('\n'+str(timestep)+','+str(SNR[SNR_counter])+','+str(sim_param.noise_sigma)+','+str(sim_param.cw_amplitude)+','+str(sim_param.cw_frequency)+','+str(sim_param.modulation_frequency)+','+str(abc_max_sum)+','+str(def_max_sum)+','+str(ghi_max_sum))
            
         three_phi_max_sum = abc_max_sum+def_max_sum+ghi_max_sum
         sum_hist.append(three_phi_max_sum)
            
         threshold_drop_flag=False
         for threshold_counter in range(0,len(threshold)):
            if((three_phi_max_sum)>threshold[threshold_counter]):
               num_passed_events[threshold_counter] += 1
            elif(threshold_drop_flag == False):
               threshold_drop_flag = True 

      plt.figure(4)
      plt.hist(np.array(first_hist),bins=25)
      plt.title("SNR: %.2f"%SNR[SNR_counter])
      plt.show()
      
      # Calculate trigger rate for current threshold
      for threshold_counter in range(0,len(threshold)):
         trigger_rate[threshold_counter] = (float(num_passed_events[threshold_counter])*float(sim_param.simulation_rate))/float(num_runs)
         graph_filename.write(str(SNR[SNR_counter])+','+str(threshold[threshold_counter])+','+str(trigger_rate[threshold_counter])+'\n')
     

      # Close Threshold Loop 
      plt.figure(1)
      plt.semilogy(threshold,trigger_rate,label='SNR: '+str(SNR[SNR_counter]))
   plt.legend()
   plt.grid(True)
   plt.xlabel("Threshold [DAC]")
   plt.ylabel("Trigger Rate [Hz]")
   if(sim_param.cw_flag):
      plt.title("TISC SNR Curves, Simulation Rate: %1.2fMHz, Event Rate: %1.2fMHz, CW Amp: %d*Noise RMS" % (sim_param.simulation_rate,sim_param.event_rate,sim_param.cw_amplitude/sim_param.noise_sigma))
   else:
      if(sim_param.boresight==0):
         if(sim_param.baseline==0):
            plt.title("Sim On Boresight, Long Baseline TISC SNR Curves, Sim Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (sim_param.simulation_rate/10**6,sim_param.event_rate/10**6))
         elif(sim_param.baseline==1):
            plt.title("Sim On Boresight, Short Baseline TISC SNR Curves, Sim Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (sim_param.simulation_rate/10**6,sim_param.event_rate/10**6))
      if(sim_param.boresight==1):
         if(sim_param.baseline==0):
            plt.title("Sim Off Boresight, Long Baseline TISC SNR Curves, Sim Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (sim_param.simulation_rate/10**6,sim_param.event_rate/10**6))
         elif(sim_param.baseline==1):
            plt.title("Sim Off Boresight, Short Baseline TISC SNR Curves, Sim Rate: %1.2fMHz, Event Rate: %1.2fMHz" % (sim_param.simulation_rate/10**6,sim_param.event_rate/10**6))
   plt.savefig(str(output_dir)+"/TISC_SNR_canvas.png")
   # Close SNR Loop

   # Save files and close
   settings_filename.write("\nEnd time: "+str(time.strftime('%Y_%m_%d_%H.%m.%S')))
   settings_filename.close()
   data_filename.close()
