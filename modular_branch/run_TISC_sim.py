#!/usr/bin/env python

if __name__ == '__main__':
   import ROOT
   import time
   import os
   from TISC_sim import TISC_sim
   from impulse import impulse_gen
   import numpy as np
   import subprocess as Popen

   num_samples = 48
   impulse_pos = 10
   signal_amp = 10
   draw_flag = 0
   b_input_delay = 3
   c_input_delay = 5
   low_SNR = 0.0001
   high_SNR = 6.0
   step_SNR = 1.0
   low_threshold = 0.0
   high_threshold = 350.0
   step_threshold = 50.0
   num_runs = 5
   upsample = 32
   digitization_factor = 1.0
   num_upsamples = upsample*num_samples
   delay_type_flag = 0

   # Write settings file
   output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/"+time.strftime('%Y_%m_%d_%H.%m.%S'))
   print output_dir
   if not os.path.exists(output_dir):
      os.makedirs(output_dir)
   settings_filename = open(output_dir+"/settings.txt","w")

   settings_filename.write("Input parameters for TISC simulation\n\n")
   settings_filename.write("Start time: "+str(time.strftime('%Y_%m_%d_%H.%m.%S')))
   settings_filename.write("\nNumber of samples per run: " + str(num_samples))
   settings_filename.write("\nImpulse Position: " + str(impulse_pos))
   settings_filename.write("\nSignal Amplitude: " + str(signal_amp))
   settings_filename.write("\nCh B Input Delay: " + str(b_input_delay))
   settings_filename.write("\nCh C Input Delay: " + str(c_input_delay))
   settings_filename.write("\nMinimum SNR: " + str(low_SNR))
   settings_filename.write("\nMaximum SNR: " + str(high_SNR))
   settings_filename.write("\nSNR Step: " + str(step_SNR))
   settings_filename.write("\nMinimum Threshold: " + str(low_threshold))
   settings_filename.write("\nMaximum Threshold: " + str(high_threshold))
   settings_filename.write("\nThreshold Step: " + str(step_threshold))
   settings_filename.write("\nNumber of Runs: " + str(num_runs))
   settings_filename.write("\nUpsample Factor: " + str(upsample))
   settings_filename.write("\nDigitization Threshold: " + str(digitization_factor)+"*Noise_RMS")
   

   # Get impulse signal
   #a_input_signal = np.zeros(num_upsamples)
   #a_input_signal = impulse_gen(num_samples,impulse_pos,upsample=upsample,draw_flag=draw_flag,output_dir=output_dir)
   SNR_hist = {}

   # Set up ROOT histogram
   SNR_canvas = ROOT.TCanvas('TISC_SNR', "TISC_SNR",200,10,700,500)
   ROOT.gROOT.Reset()
   ROOT.gStyle.SetOptStat("")
   ROOT.gStyle.SetTitleFillColor(10)
   ROOT.gStyle.SetTitleX(0.1)
   ROOT.gStyle.SetTitleW(0.8)
   #SNR_canvas.SetLogy()
   
   line_color = 1
   threshold = np.linspace(low_threshold,high_threshold,(high_threshold-low_threshold)/step_threshold+1)
   print threshold

   outfile = ROOT.TFile(output_dir+"/TISC_SNR.root","RECREATE")
   #SNR_hist[1] = ROOT.TH1D('TISC_SNR','TISC SNR Curves',high_threshold/step_threshold,low_threshold,high_threshold)
   SNR_hist[0] = ROOT.TH1D('TISC_SNR','TISC SNR Curves',len(threshold),low_threshold,high_threshold)
   SNR_hist[0].SetMinimum(-0.5)
   SNR_hist[0].SetMaximum(num_runs*1.1)
   SNR_hist[0].GetYaxis().SetTitle("Counts")
   SNR_hist[0].GetXaxis().SetTitle("Threshold")
   SNR_hist[0].Draw()
   x = np.linspace(low_threshold,high_threshold,len(threshold))
   y = num_runs/2.0+0*x
   fifty_percent_eff = ROOT.TGraph(len(x),x,y)
   fifty_percent_eff.SetLineColor(20)
   fifty_percent_eff.Draw("SAME")

   legend = ROOT.TLegend(0.6,0.6,0.8,0.8)
   legend.AddEntry(fifty_percent_eff,"50% Efficency",'l')
   legend.Draw("SAME")

   SNR = np.linspace(low_SNR,high_SNR,((high_SNR-low_SNR)/step_SNR)+1)
   print SNR
   for SNR_counter in range(0,len(SNR)):
      SNR_hist[SNR[SNR_counter]] = ROOT.TH1D('TISC_SNR_'+str(SNR[SNR_counter]),'TISC SNR Curves',len(threshold),low_threshold,high_threshold)
      print len(threshold)
      print low_threshold
      print high_threshold
      for threshold_counter in range(0,len(threshold)):
         for runNum in range(0,num_runs):
            print"SNR: "+str(SNR[SNR_counter])+"\tThreshold: "+str(threshold[threshold_counter])+"\tRun Number: "+str(runNum)
            event_passed_flag = 0
            event_passed_flag = TISC_sim(SNR[SNR_counter],threshold[threshold_counter],
                                                impulse_pos,b_input_delay,c_input_delay,upsample=upsample,
                                                num_samples=num_samples,draw_flag=draw_flag,
                                                digitization_factor=digitization_factor,
                                                delay_type_flag=delay_type_flag,output_dir=output_dir)
            if(event_passed_flag):
               SNR_hist[SNR[SNR_counter]].Fill(threshold[threshold_counter])
               print "This event passed"
            print ">>>>>>>>>>>>>>>>>"
      print "Drawing histogram for SNR "+str(SNR[SNR_counter])
      SNR_hist[SNR[SNR_counter]].SetLineColor(line_color)
      SNR_hist[SNR[SNR_counter]].Draw("*,C,SAME")
      legend.AddEntry(SNR_hist[SNR[SNR_counter]],"SNR="+str(SNR[SNR_counter]),'l')
      legend.Draw("SAME")
      line_color += 1
   #NR_canvas.SetLogy()
   SNR_canvas.Update()
   print "Done itterating over steps"
      
   ROOT.gSystem.ProcessEvents()
   img = ROOT.TImage.Create()
   img.FromPad(SNR_canvas)
   img.WriteImage(output_dir+'/TISC_SNR_canvas.png')

   outfile.WriteTObject(SNR_canvas)

   outfile.Write()
   outfile.Close()
   outfile.ls()
   settings_filename.write("/nEnd time: "+str(time.strftime('%Y_%m_%d_%H.%m.%S')))
   settings_filename.close()
   dummy = raw_input('Press any key to close')
