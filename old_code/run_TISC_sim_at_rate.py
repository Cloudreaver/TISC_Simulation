#!/usr/bin/env python

if __name__ == '__main__':
   from ROOT import TGraph, TMultiGraph,TCanvas,TImage,gROOT,gStyle,gSystem,gPad,TFile,TLegend
   import time
   import os
   from sum_corr import sum_correlation
   from impulse import impulse_gen
   import numpy as np
   import subprocess as Popen
   from array import array
   
   #############################################################
   # Parameters
   num_runs = 3000  # Number of events to generate per loop
   upsample = 32 # Upsamping factor
   digitization_factor = 1.0 # Digitization factor( *Noise_RMS)
   max_trigger_rate = 162500000.0 # TISC Clock Speed
   event_rate = 3.0*max_trigger_rate # Rate to generate events
   num_samples = 48 # Length of Signal Window
   #signal_amp = 10  
   draw_flag = 0    # 1=draw event graphs, 0=don't
   SNR_draw_flag =1 # 1=draw SNR graphs, 0=don't

   impulse_pos = 10 # Position of impulse in Ch A
   b_input_delay = 3# Ch B signal offset
   c_input_delay = 5# Ch C signal offset

   low_SNR = 0.0001 # Lowest SNR
   high_SNR = 6.0   # Highest SNR
   step_SNR = 1.0   # SNR interval

   low_threshold = 0.0    # Lowest Threshold
   high_threshold = 600.0 # Highest Threshold
   step_threshold = 25.0 # Threshold Interval

   #low_event_rate = 1.0
   #high_event_rate = 200000000.0
   #num_event_rate = 10
   #############################################################
   
   # Set some variables
   time_since_last_trigger = float(0.0) #Time since last trigger
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
   output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/"+time.strftime('%Y_%m_%d_%H.%m.%S'))
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   settings_filename = open(output_dir+"/settings.txt","w")

   settings_filename.write("Input parameters for TISC simulation\n\n")
   settings_filename.write("Start time: "+str(time.strftime('%Y_%m_%d_%H.%m.%S')))
   settings_filename.write("\nNumber of samples per run: " + str(num_samples))
   settings_filename.write("\nImpulse Position: " + str(impulse_pos))
   #settings_filename.write("\nSignal Amplitude: " + str(signal_amp))
   settings_filename.write("\nCh B Input Delay: " + str(b_input_delay))
   settings_filename.write("\nCh C Input Delay: " + str(c_input_delay))
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
   a_input_signal = np.zeros(num_upsamples)
   a_input_signal = impulse_gen(num_samples,impulse_pos,upsample=upsample,
                                draw_flag=draw_flag,output_dir=output_dir)

   # Set up ROOT Stuff
   if(SNR_draw_flag):
      SNR_canvas = TCanvas('TISC_SNR', "TISC_SNR",800,600)
      SNR_canvas.SetLogy()
   multigraph = TMultiGraph('SNR_Graph','SNR_Graph')
   gROOT.Reset()
   gStyle.SetOptStat("")
   gStyle.SetTitleFillColor(10)
   gStyle.SetTitleX(0.1)
   gStyle.SetTitleW(0.8)
   
   outfile = TFile(output_dir+"/TISC_SNR.root","RECREATE")
   #SNR_hist[1] = ROOT.TH1D('TISC_SNR','TISC SNR Curves',high_threshold/step_threshold,low_threshold,high_threshold)
   #SNR_hist[0] = ROOT.TH1D('TISC_SNR','TISC SNR Curves',len(threshold),low_threshold,high_threshold)
   #SNR_hist[0].SetMinimum(-0.5)
   #SNR_hist[0].SetMaximum(num_runs*1.1)
   #SNR_hist[0].GetYaxis().SetTitle("Counts")
   #SNR_hist[0].GetXaxis().SetTitle("Threshold")
   #SNR_hist[0].Draw()
   x = np.linspace(low_threshold,high_threshold,len(threshold))
   y = max_trigger_rate+0*x
   max_trigger_rate_graph = TGraph(len(x),x,y)
   max_trigger_rate_graph.SetLineColor(20)
   max_trigger_rate_graph.SetName("Maximum Trigger Rate")
   max_trigger_rate_graph.SetTitle("Maximum Trigger Rate")
   multigraph.Add(max_trigger_rate_graph)
   max_trigger_rate_graph.Write()

   z = event_rate+0*x
   event_rate_graph = TGraph(len(x),x,z)
   event_rate_graph.SetLineColor(46)
   event_rate_graph.SetName("Event Rate")
   event_rate_graph.SetTitle("Event Rate")
   multigraph.Add(event_rate_graph)
   event_rate_graph.Write()

   #legend = ROOT.TLegend(0.6,0.6,0.8,0.8)
   #legend.AddEntry(fifty_percent_eff,"50% Efficency",'l')
   #legend.Draw("SAME")

   
   # Start Loop over SNR values
   for SNR_counter in range(0,len(SNR)):
      print "Starting SNR of "+str(SNR[SNR_counter])

      # Start loop over threshold values
      for threshold_counter in range(0,len(threshold)):
         # Reset values
         trigger_rate[threshold_counter] = 0 
         time_since_last_trigger = 2.0*(1.0/event_rate)
         num_passed_events[threshold_counter] = 0

         # Start loop over events
         for runNum in range(0,num_runs):
            event_passed_flag = 0 # Reset value
            #print "Time since last trigger: "+str(time_since_last_trigger)
            # Check to see if trigger is available
            if (time_since_last_trigger >= (1.0/max_trigger_rate)):
               # Run GLITC code
               event_passed_flag = sum_correlation(SNR[SNR_counter],threshold[threshold_counter],a_input_signal,
                                                   b_input_delay,c_input_delay,upsample=upsample,
                                                   num_samples=num_samples,draw_flag=draw_flag,
                                                   digitization_factor=digitization_factor,output_dir=output_dir)
               # Check to see if event passed
               if(event_passed_flag):
                  num_passed_events[threshold_counter] += 1
                  #print "This event passed"
                  time_since_last_trigger = 1.0/event_rate
            else:
               time_since_last_trigger = time_since_last_trigger + 1.0/event_rate
               #print "Trigger is busy"
            #print ">>>>>>>>>>>>>>>>>"
         # Close of Event Loop

         # Calculate trigger rate for current threshold
         #print float(num_passed_events[threshold_counter])
         #print float(num_runs)
         #print event_rate
         trigger_rate[threshold_counter] =(float(num_passed_events[threshold_counter])/float(num_runs))*event_rate
         #print trigger_rate[threshold_counter]
      # Close Threshold Loop 
      
      # Make all the Graphs
      SNR_Graph[SNR_counter] = TGraph(len(threshold),array('d',threshold),array('d',trigger_rate))
      SNR_Graph[SNR_counter].SetName('SNR '+str(SNR[SNR_counter]))
      SNR_Graph[SNR_counter].SetTitle('SNR '+str(SNR[SNR_counter]))
      SNR_Graph[SNR_counter].SetLineColor(line_color)
      SNR_Graph[SNR_counter].SetLineWidth(2)
      SNR_Graph[SNR_counter].SetFillStyle(0)
      SNR_Graph[SNR_counter].SetDrawOption("A*")
      multigraph.Add(SNR_Graph[SNR_counter])
      SNR_Graph[SNR_counter].Write()
      line_color += 1
   # Close SNR Loop

   # Set up Multigraph and add graphs
   multigraph.SetTitle("TISC SNR Curves (@ "+str(event_rate/(10**6))+" MHz Event Rate); Threshold [unitless]; Averaged Trigger Rate [Hz]")
   if(SNR_draw_flag):
      multigraph.Draw("ACP")
      SNR_canvas.BuildLegend(0.8,0.7,0.99,0.9)
      gPad.Update()
      gPad.Modified()
   
   print "Done itterating over steps"
   # Draw Everything
   if(SNR_draw_flag):   
      gSystem.ProcessEvents()
      img = TImage.Create()
      img.FromPad(SNR_canvas)
      img.WriteImage(output_dir+'/TISC_SNR_canvas.png')

      outfile.WriteTObject(SNR_canvas)

   # Save files and close
   outfile.Write()
   outfile.Close()
   #outfile.ls()
   settings_filename.write("\nEnd time: "+str(time.strftime('%Y_%m_%d_%H.%m.%S')))
   settings_filename.close()
   if(SNR_draw_flag):
      dummy = raw_input('Press any key to close')
