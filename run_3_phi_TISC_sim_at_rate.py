#!/usr/bin/env python

if __name__ == '__main__':
   import time
   import os
   from three_phi_sector_TISC_sim import TISC_sim
   import numpy as np
   import subprocess as Popen
   from array import array
   from simulation_parameters import sim_param
   
   # This is so that it will run on the cluster
   if(sim_param.debug==False):    
      import matplotlib
      matplotlib.use('Agg')
   
   import matplotlib.pyplot as plt
   
  
   
   ######### Set some variables ###########
   num_runs = int(sim_param.num_events*(sim_param.simulation_rate/sim_param.event_rate)) 
   # Number of upsampled samples
   num_upsamples = sim_param.upsample*sim_param.num_samples 
   line_color = 1
   
   # Generate array of thresholds
   threshold = np.linspace(sim_param.low_threshold,sim_param.high_threshold,
                           (sim_param.high_threshold-sim_param.low_threshold)
                           /sim_param.step_threshold+1) 
   num_passed_events = np.zeros(len(threshold))
   trigger_rate = np.zeros(len(threshold))
   SNR = np.linspace(sim_param.low_SNR,sim_param.high_SNR,((sim_param.high_SNR-sim_param.low_SNR)/sim_param.step_SNR)+1)
   num_delays = [63,46]
   #########################################
   
   ###### Phi sectors have alternating baselines #########
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
   #######################################################      
         
   ####### Write settings file #########
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

   graph_filename = open(output_dir+"/graph_output.dat","w")
   graph_filename.write("SNR,threshold,trigger_rate\n")
   data_filename = open(output_dir+"/data_output.dat","w")
   data_filename.write("time_step,SNR,thermal_noise_sigma,cw_amplitude,carrier_frequency,modulation_frequency,abc_max,abc_as_max,abc_correlation_mean,def_max,def_as_max,def_correlation_mean,ghi_max,ghi_as_max,ghi_correlation_mean")
   ###########################################3
   
   ########## Start Loop over SNR values ###################
   for SNR_counter in range(0,len(SNR)):
      first_hist = []
      second_hist = []
      sum_hist = []
            
      # Reset values
      trigger_rate[threshold_counter] = 0 
      time_since_last_trigger = 2.0*(1.0/sim_param.event_rate)
      num_passed_events = np.zeros(len(threshold))
      
      ################# Start loop over events ####################
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
            
         # Add three phi sectors together
         three_phi_max_sum = abc_max_sum+def_max_sum+ghi_max_sum
         sum_hist.append(three_phi_max_sum)
            
         threshold_drop_flag=False
         for threshold_counter in range(0,len(threshold)):
            if((three_phi_max_sum)>threshold[threshold_counter]):
               num_passed_events[threshold_counter] += 1
            elif(threshold_drop_flag == False):
               threshold_drop_flag = True 
      ##### Close timestep loop ###############################33
      """
      plt.figure(4)
      plt.hist(np.array(first_hist),bins=25)
      plt.title("SNR: %.2f"%SNR[SNR_counter])
      plt.show()
      """
      # Calculate trigger rate for current threshold
      for threshold_counter in range(0,len(threshold)):
         trigger_rate[threshold_counter] = (float(num_passed_events[threshold_counter])*float(sim_param.simulation_rate))/float(num_runs)
         graph_filename.write(str(SNR[SNR_counter])+','+str(threshold[threshold_counter])+','+str(trigger_rate[threshold_counter])+'\n')
      
      
      # Add individual SNR plots
      plt.figure(1)
      plt.semilogy(threshold,trigger_rate,label='SNR: '+str(SNR[SNR_counter]))
      
   ######## Close SNR Loop #################################3
      
   ####### SNR Curve details ###############################
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
   #########################################################

   # Save files and close
   settings_filename.write("\nEnd time: "+str(time.strftime('%Y_%m_%d_%H.%m.%S')))
   settings_filename.close()
   data_filename.close()
