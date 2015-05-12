#!/usr/bin/env python

if __name__ == '__main__':
   from ROOT import TGraph, TMultiGraph,TCanvas,TImage,gROOT,gStyle,gSystem,gPad,TFile,TLegend, TH1F, TTree
   import time
   import os
   from TISC_sim import TISC_sim
   #from impulse import impulse_gen
   import numpy as np
   import subprocess as Popen
   from array import array
   
   #############################################################
   # Parameters
   num_events = 100000             # Number of events to generate per loop
   upsample = 1                     # Upsamping factor
   digitization_factor = 1.0        # Digitization factor( *Noise_RMS)
   num_bits = 3                     # Number of bits available to digitizer
   simulation_rate = 100.0          # Simulation Rate
   event_rate=100.0                 # Rate to generate events
   num_runs = int(num_events*(simulation_rate/event_rate)) 
   num_samples = 74                 # Length of Signal Window
   noise_sigma = 20.0
   noise_mean = 0.0
   draw_flag = False                # 1=draw event graphs, 0=don't                        
   SNR_draw_flag = True             # 1=draw SNR graphs, 0=don't
   delay_type_flag = 1              # 1=use GLITC delays, 0=all possible delays
   cw_flag = True
   carrier_frequency=260000000.0    # Hz
   modulation_frequency=1.0  # Hz
   cw_factor = 2.0
   peak_amplitude = cw_factor*noise_sigma #Peak amplitude in mV
   

   impulse_pos = 5                  # Position of impulse in Ch A (must be contained within first 32 samples)
   b_input_delay = 0                # Ch B signal offset
   c_input_delay = 0                # Ch C signal offset

   low_SNR = 0.0                    # Lowest SNR
   high_SNR = 5.0                   # Highest SNR
   step_SNR = 1.0                 # SNR interval

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

   # Write settings file
   if(cw_flag):
      output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/TISC_SIM_CW"+str(cw_factor)+" Num"+str(num_runs)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
   else:
      output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/TISC_SIM_"+"Num"+str(num_runs)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   settings_filename = open(output_dir+"/settings.txt","w")
   #print output_dir
   settings_filename.write("Input parameters for TISC simulation\n\n")
   settings_filename.write("Start time: "+str(time.strftime('%Y_%m_%d_%H.%M.%S')))
   settings_filename.write("\nNumber of events per run: " + str(num_runs))
   settings_filename.write("\nNumber of samples per event: " + str(num_samples))
   settings_filename.write("\nNumber of bits: "+ str(num_bits))
   settings_filename.write("\nImpulse Position: " + str(impulse_pos))
   if(delay_type_flag):
      settings_filename.write("\nUsing GLITC Delays")
   else:
      settings_filename.write("\nUsing all possible event delays")
   if(cw_flag):
      settings_filename.write("\nAdding CW Noise")
      settings_filename.write("\nCW Frequency: " +str(carrier_frequency))
      settings_filename.write("\nModulation Frequency: " + str(modulation_frequency))
      settings_filename.write("\nCW Amplitude: " + str(peak_amplitude))
   
   settings_filename.write("\nCh B Input Delay: " + str(b_input_delay))
   settings_filename.write("\nCh C Input Delay: " + str(c_input_delay))
   settings_filename.write("\nNoise Mean: " +str(noise_mean))
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
   

   # Get impulse signal
   #a_input_signal = np.zeros(num_upsamples)
   #a_input_signal = impulse_gen(num_samples,impulse_pos,upsample=upsample,draw_flag=draw_flag,output_dir=output_dir)

   # Set up ROOT Stuff
   gROOT.Reset()
   gStyle.SetOptStat("")
   gStyle.SetTitleFillColor(10)
   gStyle.SetTitleX(0.1)
   gStyle.SetTitleW(0.8)
   gROOT.SetBatch()
   
   
   
   if(SNR_draw_flag):
      SNR_canvas = TCanvas('TISC_SNR', "TISC_SNR",1600,1200)
      SNR_canvas.SetLogy()
   multigraph = TMultiGraph('SNR_Graph','SNR_Graph')
   
   
   SNR_counter = 0
   threshold_counter = 0
   #tree.Branch('SNR',SNR[SNR_counter])
   #tree.Branch('threshold',threshold[threshold_counter])
   #tree.Branch('run_number',runNum)
   #

   data_filename = open(output_dir+"/output.dat","w")
   
   #threshold_for_branch = np.zeros(1)
   # Start Loop over SNR values
   for SNR_counter in range(0,len(SNR)):
      #print "\nStarting SNR of "+str(SNR[SNR_counter])
      SNR_Hist[SNR_counter] = TH1F('SNR_'+str(SNR[SNR_counter]),'SNR_'+str(SNR[SNR_counter]),len(threshold),low_threshold,high_threshold)
      #tree.Branch('SNR_'+str(SNR[SNR_counter]),threshold_for_branch,'SNR_'+str(SNR[SNR_counter])+'/D')
      
      # Start loop over threshold values
      
         # Reset values
      trigger_rate[threshold_counter] = 0 
      time_since_last_trigger = 2.0*(1.0/event_rate)
      num_passed_events = np.zeros(len(threshold))
      

      # Start loop over events
      for timestep in range(0,num_runs):
         event_passed_flag = 0 # Reset value
         max_sum = 0

         # Check to see if trigger is available
         if ((timestep % int((simulation_rate/event_rate)) == 0)):
            #print "Generating impulsive event"
            event_passed_flag, max_sum = TISC_sim(SNR[SNR_counter],100,
                                         impulse_pos,b_input_delay,c_input_delay,num_bits=num_bits,delay_type_flag=delay_type_flag,upsample=upsample,num_samples=num_samples,
                                         cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
                                         noise_sigma=noise_sigma,noise_mean=noise_mean,
                                         draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir=output_dir)
         else:
            #print "Generating Noise Event"
            event_passed_flag, max_sum = TISC_sim(0.0,100,
                                         impulse_pos,b_input_delay,c_input_delay,num_bits=num_bits,delay_type_flag=delay_type_flag,upsample=upsample,num_samples=num_samples,
                                         cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
                                         noise_sigma=noise_sigma,noise_mean=noise_mean,
                                         draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir=output_dir)
      
         for threshold_counter in range(0,len(threshold)):
            if(max_sum>threshold[threshold_counter]):
               #threshold_for_branch[0]=threshold[threshold_counter]
               num_passed_events[threshold_counter] += 1
               SNR_Hist[SNR_counter].Fill(threshold[threshold_counter])
               #print num_passed_events[threshold_counter]


      # Calculate trigger rate for current threshold
      for threshold_counter in range(0,len(threshold)):
         trigger_rate[threshold_counter] =(float(num_passed_events[threshold_counter])/float(num_runs))*event_rate
         data_filename.write(str(SNR[SNR_counter])+'\t'+str(threshold[threshold_counter])+'\t'+str(trigger_rate[threshold_counter])+'\n')
         #print str(SNR[SNR_counter])+'\t'+str(threshold[threshold_counter])+'\t'+str(trigger_rate[threshold_counter])

     

      # Close Threshold Loop 
      
      
      #tree.Fill()
      
      # Make all the Graphs   
      #SNR_Hist[SNR_counter].Write()
      SNR_Graph[SNR_counter] = TGraph(len(threshold),array('d',threshold),array('d',trigger_rate))
      SNR_Graph[SNR_counter].SetName('SNR '+str(SNR[SNR_counter])+"_Graph")
      SNR_Graph[SNR_counter].SetTitle('SNR '+str(SNR[SNR_counter]))
      SNR_Graph[SNR_counter].SetLineColor(line_color)
      SNR_Graph[SNR_counter].SetLineWidth(2)
      SNR_Graph[SNR_counter].SetFillStyle(0)
      SNR_Graph[SNR_counter].SetDrawOption("A*")
      multigraph.Add(SNR_Graph[SNR_counter])
      #SNR_Graph[SNR_counter].Write()
      line_color += 1
   # Close SNR Loop

   # Set up Multigraph and add graphs
   if (cw_flag):
      multigraph.SetTitle("TISC Sim "+str(event_rate)+" MHz Event Rate "+str(simulation_rate)+" Simulation Rate w/ CW @"+str(cw_factor))
   else:
      multigraph.SetTitle("TISC Sim "+str(event_rate)+" MHz Event Rate "+str(simulation_rate)+" Simulation Rate")
   if(SNR_draw_flag):
      multigraph.Draw("AC")
      SNR_canvas.BuildLegend(0.7,0.6,0.99,0.9)
      gPad.Update()
      gPad.Modified()
   
   #print "Done itterating over steps"
   # Draw Everything
   if(SNR_draw_flag):   
      gSystem.ProcessEvents()
      img = TImage.Create()
      img.FromPad(SNR_canvas)
      img.WriteImage(output_dir+'/TISC_SNR_canvas.png')

      #outfile.WriteTObject(SNR_canvas)

   # Save files and close
   #outfile.Write()
   #outfile.Close()
   #outfile.ls()
   settings_filename.write("\nEnd time: "+str(time.strftime('%Y_%m_%d_%H.%m.%S')))
   settings_filename.close()
   data_filename.close()
   #if(SNR_draw_flag):
      #dummy = raw_input('Press any key to close')
