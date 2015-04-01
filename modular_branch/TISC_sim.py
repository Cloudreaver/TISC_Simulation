#!/usr/bin/env python

# A very simple impulsive signal + noise generator to get the 

import random
from ROOT import TTree, TBranch, gROOT,gSystem,TImage, TStyle, TFile, gDirectory, TH1D, TCanvas, gStyle, TGraph
from array import array
import numpy as np
from impulse import impulse_gen, butter_bandpass_filter, butter_bandpass
from noise import generate_noise
from digitizer import digitize
from sum_correlator import sum_correlate
from cw import generate_cw

def TISC_sim(SNR,threshold,
             impulse_position,b_input_delay,c_input_delay,num_bits=3,
             noise_sigma=20.0,noise_mean=0.0,
             sample_freq=2600000000.0,TISC_sample_length=16,
             num_samples=74,upsample=1,cw_flag=0,
             peak_amplitude=25.0,carrier_frequency=260000000.0,modulation_frequency=16000000.0,
             seed=5522684,draw_flag=0,digitization_factor=1,
             delay_type_flag=0,
             output_dir="output/"):
   
   gROOT.Reset()
   gStyle.SetOptStat("")
   gStyle.SetTitleFillColor(10)
   gStyle.SetTitleX(0.1)
   gStyle.SetTitleW(0.8)

   
                           

   
   # Setup
   save_output_flag = 0
   if(save_output_flag):
      outfile = str(output_dir+"/test.root")
   trigger_flag = 0
   #num_bits = 3 # Number of bits available to the digitizer
   num_upsamples = num_samples*upsample
   filter_flag = True
   #sample_frequency = 2800000000.0
   
   # Fill numpy arrays with zeros
   a_input_noise = np.zeros(num_upsamples)
   b_input_noise = np.zeros(num_upsamples)
   c_input_noise = np.zeros(num_upsamples)
   time = np.zeros(num_samples)
   upsampled_time = np.zeros(num_upsamples)
   a_input_signal = np.zeros(num_upsamples)
   b_input_signal = np.zeros(num_upsamples)
   c_input_signal = np.zeros(num_upsamples)
   a_input_signal_noise = np.zeros(num_upsamples)
   b_input_signal_noise = np.zeros(num_upsamples)
   c_input_signal_noise = np.zeros(num_upsamples)
   a_dig_waveform = np.zeros(num_samples)
   b_dig_waveform = np.zeros(num_samples)
   c_dig_waveform = np.zeros(num_samples)
   empty_list = np.zeros(num_upsamples)

   if(save_output_flag):
      # Set up ROOT file/tree
      f = TFile(outfile,"recreate")
      rf_tree = TTree("input_rf", "RF Tree")
      rf_tree.Branch('time',time,'time['+str(num_samples)+']/D')
      rf_tree.Branch('Ch_A_Noise',a_input_noise,'AinputNoise['+str(num_upsamples)+']/D')
      rf_tree.Branch('Ch_B_Noise',b_input_noise,'BinputNoise['+str(num_upsamples)+']/D')
      rf_tree.Branch('Ch_C_Noise',c_input_noise,'CinputNoise['+str(num_upsamples)+']/D')
      
      rf_tree.Branch('Ch_A_Signal',a_input_signal,'AinputSignal['+str(num_upsamples)+']/D')
      rf_tree.Branch('Ch_B_Signal',b_input_signal,'BinputSignal['+str(num_upsamples)+']/D')
      rf_tree.Branch('Ch_C_Signal',c_input_signal,'CinputSignal['+str(num_upsamples)+']/D')
      
      rf_tree.Branch('Ch_A_SigNos',a_input_signal_noise,'AinputSignalNoise['+str(num_upsamples)+']/D')
      rf_tree.Branch('Ch_B_SigNos',b_input_signal_noise,'BinputSignalNoise['+str(num_upsamples)+']/D')
      rf_tree.Branch('Ch_C_SigNos',c_input_signal_noise,'CinputSignalNoise['+str(num_upsamples)+']/D')
      
      rf_tree.Branch('Ch_A_3Bit',a_dig_waveform,'AdigWaveform['+str(num_samples)+']/D')
      rf_tree.Branch('Ch_B_3Bit',b_dig_waveform,'BdigWaveform['+str(num_samples)+']/D')
      rf_tree.Branch('Ch_C_3Bit',c_dig_waveform,'CdigWaveform['+str(num_samples)+']/D')
   
   
   
   
   
   for i in range(0,num_upsamples):
      upsampled_time[i] = (i/(sample_freq*upsample))*(10**9)

   for i in range(0,num_samples):
      time[i] = (i/(sample_freq))*(10**9)

###################################
   # Generate Thermal Noise
   a_input_noise = generate_noise(num_samples,upsample,noise_mean,noise_sigma,filter_flag)
   b_input_noise = generate_noise(num_samples,upsample,noise_mean,noise_sigma,filter_flag)
   c_input_noise = generate_noise(num_samples,upsample,noise_mean,noise_sigma,filter_flag)
###################################


#####################################
   # Determine RMS of noise and signal amplitude

   noise_rms = np.sqrt(np.mean((a_input_noise-noise_mean)**2,))
   
   signal_amp = SNR*2*noise_rms

#####################################
   
   
#################################
   #Generate CW noise if desired
   if cw_flag:
      a_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      a_input_noise += a_input_cw_noise

      b_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      b_input_noise += b_input_cw_noise

      c_input_cw_noise = generate_cw(num_samples,upsample,sample_freq,carrier_frequency,modulation_frequency,peak_amplitude,filter_flag)
      c_input_noise += c_input_cw_noise

#####################################

 
#####################################
   # Generate impulse
   if (SNR != 0):
      # Generate Signal and Amplify
      a_input_signal = impulse_gen(num_samples,impulse_position,upsample,draw_flag=draw_flag,output_dir=output_dir)
      difference=np.amax(a_input_signal)-np.amin(a_input_signal) # Get peak to peak voltage
      a_input_signal *= (1/difference) # Normalize input
      a_input_signal *= signal_amp # Amplify
      #print "SNR: " +str(SNR)
      #print "Noise RMS: " +str(noise_rms)
      #print "Signal Amp: " +str(signal_amp)
      #print "Difference: "+str(difference)
      #print "New signal amp: "+str(np.amax(a_input_signal)-np.amin(a_input_signal))
      b_input_signal = np.concatenate([empty_list[:b_input_delay*upsample],a_input_signal[:num_upsamples-b_input_delay*upsample]])
      c_input_signal = np.concatenate([empty_list[:c_input_delay*upsample],a_input_signal[:num_upsamples-c_input_delay*upsample]])

      # Add the signal to the noise
      a_input_signal_noise = np.add(a_input_noise, a_input_signal)
      b_input_signal_noise = np.add(b_input_noise, b_input_signal)
      c_input_signal_noise = np.add(c_input_noise, c_input_signal)
   else:
      a_input_signal_noise = a_input_noise
      b_input_signal_noise = b_input_noise
      c_input_signal_noise = c_input_noise
##########################################

##########################################
   # Digitized the incoming signal and noise (RITC)
   a_dig_waveform = digitize(a_input_signal_noise,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)
   b_dig_waveform = digitize(b_input_signal_noise,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)
   c_dig_waveform = digitize(c_input_signal_noise,num_samples,upsample,num_bits,noise_mean,noise_rms,digitization_factor)

##########################################

##########################################
   # Run the signal through the GLITC module to get trigger
   trigger_flag, max_sum = sum_correlate(num_samples,a_dig_waveform,b_dig_waveform,c_dig_waveform,threshold,TISC_sample_length,delay_type_flag=delay_type_flag)

#########################################

#########################################
   # Output data
   if(save_output_flag):
      # Now to more ROOT stuff
      rf_tree.Fill()
      f.Write()
      f.Close()

   if draw_flag:
      c1 = TCanvas('c1','Signal Simulation',1600,900)
      #c2 = TCanvas('c2','Test',800,600)
      #c1.cd()
      c1.Divide(3,4,0,0)
      #c1.cd(1)
      #c2.cd()
      #testGraph = TGraph(num_samples,array('d',time),array('d',c_dig_waveform))
      #testGraph.GetYaxis().SetTitle("Test [ADC Counts]")
      #testGraph.GetXaxis().SetTitle("Time Bins[ns]")
      #testGraph.SetTitle("Test Graph")
      #testGraph.Draw("AL")
      
      c1.cd(1)
      a_input_signal_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',a_input_signal))
      a_input_signal_graph.GetYaxis().SetTitle("Ch A In Signal [mV]")
      a_input_signal_graph.GetXaxis().SetTitle("Time Bins [ns]")
      a_input_signal_graph.SetTitle('Ch A Signal')
   
      b_input_signal_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',b_input_signal))
      b_input_signal_graph.GetYaxis().SetTitle("Ch B In Signal [mV]")
      b_input_signal_graph.GetXaxis().SetTitle("Time Bins [ns]")
      b_input_signal_graph.SetTitle('Ch B Signal')
      
      c_input_signal_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',c_input_signal))
      c_input_signal_graph.GetYaxis().SetTitle("Ch C In Signal [mV]")
      c_input_signal_graph.GetXaxis().SetTitle('Time Bins [ns]')
      c_input_signal_graph.SetTitle('Ch C Signal')

      c2 = TCanvas('c2','Simulated Thermal Noise',1600,900)
      c2.cd()
      a_input_noise_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',a_input_noise))
      a_input_noise_graph.GetYaxis().SetTitle("Amplitude")
      a_input_noise_graph.GetXaxis().SetTitle("Time Bins [ns]")
      a_input_noise_graph.SetTitle('Thermal Noise')
      a_input_noise_graph.Draw()
      gSystem.ProcessEvents()
      img = TImage.Create()
      img.FromPad(c2)
      img.WriteImage(output_dir+"thermal_noise.png")
      
      b_input_noise_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',b_input_noise))
      b_input_noise_graph.GetYaxis().SetTitle("Ch B In Noise")
      b_input_noise_graph.GetXaxis().SetTitle("Time Bins [ns]")
      b_input_noise_graph.SetTitle('Ch B Noise')
   
      c_input_noise_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',c_input_noise))
      c_input_noise_graph.GetYaxis().SetTitle("Ch C In Noise")
      c_input_noise_graph.GetXaxis().SetTitle("Time Bins [ns]")
      c_input_noise_graph.SetTitle('Ch C Noise')
   
   
      c3 = TCanvas("c3",'Signal and Noise',1600,900)
      c3.cd()
      a_input_signal_noise_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',a_input_signal_noise))
      a_input_signal_noise_graph.GetYaxis().SetTitle("Ch A S&N [mV]")
      a_input_signal_noise_graph.GetXaxis().SetTitle("Time Bins [ns]")
      a_input_signal_noise_graph.SetTitle('Ch A Signal + Noise')
      a_input_signal_noise_graph.Draw()
      gSystem.ProcessEvents()
      img = TImage.Create()
      img.FromPad(c3)
      img.WriteImage(output_dir+"signal_noise.png")



      #c1.cd(8)
      b_input_signal_noise_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',b_input_signal_noise))
      b_input_signal_noise_graph.GetYaxis().SetTitle("Ch B S&N [mV]")
      b_input_signal_noise_graph.GetXaxis().SetTitle("Time Bins [ns]")
      b_input_signal_noise_graph.SetTitle('Ch B Signal + Noise')
   
      #c1.cd(9)
      c_input_signal_noise_graph = TGraph(num_upsamples,array('d',upsampled_time),array('d',c_input_signal_noise))
      c_input_signal_noise_graph.GetYaxis().SetTitle("Ch C S&N [mV]")
      c_input_signal_noise_graph.GetXaxis().SetTitle("Time Bins [ns]")
      c_input_signal_noise_graph.SetTitle('Ch C Signal + Noise')
   

      c4 = TCanvas('c4','3-bit Digitized Waveform',1600,900)
      c4.cd()
      a_dig_waveform_graph = TGraph(num_samples,array('d',time),array('d',a_dig_waveform))
      a_dig_waveform_graph.GetYaxis().SetTitle("Ch A 3-bit wf [ADC counts]")
      a_dig_waveform_graph.GetXaxis().SetTitle("Time Bins [ns]")
      a_dig_waveform_graph.SetTitle("Ch A RITC Output")
      a_dig_waveform_graph.Draw()
      gSystem.ProcessEvents()
      img = TImage.Create()
      img.FromPad(c4)
      img.WriteImage(output_dir+"three_bit_dig.png")



      #c1.cd(11)
      b_dig_waveform_graph = TGraph(num_samples,array('d',time),array('d',b_dig_waveform))
      b_dig_waveform_graph.GetYaxis().SetTitle("Ch B 3-bit wf [ADC counts]")
      b_dig_waveform_graph.GetXaxis().SetTitle("Time Bins [ns]")
      b_dig_waveform_graph.SetTitle("Ch B RITC Output")

      #c1.cd(12)
      c_dig_waveform_graph = TGraph(num_samples,array('d',time),array('d',c_dig_waveform))
      c_dig_waveform_graph.GetYaxis().SetTitle("Ch C 3-bit wf [ADC counts]")
      c_dig_waveform_graph.GetXaxis().SetTitle("Time Bins [ns]")
      c_dig_waveform_graph.SetTitle("Ch C RITC Output")

      #c1.cd(13)
      #abc_sum_waveform_graph = TGraph(num_samples,array('d',time),array('d',best_add_ABC))
      #abc_sum_waveform_graph.GetYaxis().SetTitle("Summed Waveform [ADC counts]")
      #abc_sum_waveform_graph.GetXaxis().SetTitle("Time Bins [ns]")
      #abc_sum_waveform_graph.SetTitle("Summed Waveform")


      #c2.cd()
      c1.cd(1)
      a_input_signal_graph.Draw("AL")
      #a_input_signal_graph.SetMinimum(-200)
      #a_input_signal_graph.SetMaximum(200)
      #a_input_signal_graph.Write()
   
      c1.cd(2)
      b_input_signal_graph.Draw("AL")
      #b_input_signal_graph.SetMinimum(-200)
      #b_input_signal_graph.SetMaximum(200)
      #b_input_signal_graph.Write()

      c1.cd(3)
      c_input_signal_graph.Draw("AL")
      #c_input_signal_graph.SetMinimum(-200)
      #c_input_signal_graph.SetMaximum(200)
      #c_input_signal_graph.Write()
   
      ######################################
      c1.cd(4)
      a_input_noise_graph.Draw("AL")
      a_input_noise_graph.SetMinimum(noise_mean-3*noise_sigma)
      a_input_noise_graph.SetMaximum(noise_mean+3*noise_sigma)
      #a_input_noise_graph.Write()

      c1.cd(5)
      b_input_noise_graph.Draw("AL")
      b_input_noise_graph.SetMinimum(noise_mean-3*noise_sigma)
      b_input_noise_graph.SetMaximum(noise_mean+3*noise_sigma)
      #b_input_noise_graph.Write()

      c1.cd(6)
      c_input_noise_graph.Draw("AL")
      c_input_noise_graph.SetMinimum(noise_mean-3*noise_sigma)
      c_input_noise_graph.SetMaximum(noise_mean+3*noise_sigma)
      #c_input_noise_graph.Write()
      ####################################
      
      c1.cd(7)
      a_input_signal_noise_graph.Draw('AL')
      a_input_signal_noise_graph.SetMinimum(noise_mean-3*noise_sigma)
      a_input_signal_noise_graph.SetMaximum(noise_mean+3*noise_sigma)
      #a_input_signal_noise_graph.Write()
   
      c1.cd(8)
      b_input_signal_noise_graph.Draw('AL')
      b_input_signal_noise_graph.SetMinimum(noise_mean-3*noise_sigma)
      b_input_signal_noise_graph.SetMaximum(noise_mean+3*noise_sigma)
      #b_input_signal_noise_graph.Write()
   
      c1.cd(9)
      c_input_signal_noise_graph.Draw('AL')
      c_input_signal_noise_graph.SetMinimum(noise_mean-3*noise_sigma)
      c_input_signal_noise_graph.SetMaximum(noise_mean+3*noise_sigma)
      #c_input_signal_noise_graph.Write()
      ########################################

      c1.cd(10)
      a_dig_waveform_graph.Draw('AL')
      a_dig_waveform_graph.SetMinimum(-(num_bits-1)**2-1)
      a_dig_waveform_graph.SetMaximum((num_bits-1)**2+1)
      #a_dig_waveform_graph.Write()

      c1.cd(11)
      b_dig_waveform_graph.Draw('AL')
      b_dig_waveform_graph.SetMinimum(-(num_bits-1)**2-1)
      b_dig_waveform_graph.SetMaximum((num_bits-1)**2+1)
      #b_dig_waveform_graph.Write()

      c1.cd(12)
      c_dig_waveform_graph.Draw('AL')
      c_dig_waveform_graph.SetMinimum(-(num_bits-1)**2-1)
      c_dig_waveform_graph.SetMaximum((num_bits-1)**2+1)
      #c_dig_waveform_graph.Write()

      #c1.cd(13)
      #abc_sum_waveform_graph.Draw('AL')
      c1.Update()
      #dummy = raw_input('Press any key to close')

   return trigger_flag

if __name__ == '__main__':
   import ROOT
   import time
   num_samples = 74
   impulse_position = 20
   num_bits = 3
   #signal_amp = 10.0
   draw_flag = True
   b_input_delay = 0
   c_input_delay = 0
   SNR = 5
   threshold = 100
   #num_runs = 100
   upsample = 1
   cw_flag = False
   cw_amplitude = 200.0
   cw_frequency = 260000000.0
   delay_type_flag = 1
   digitization_factor=1.0
   
   print "SNR: "+str(SNR)
   print "Threshold: "+str(threshold)


   # Get impulse signal
   #a_input_signal = np.zeros(num_samples*upsample)
   #a_input_signal = impulse_gen(num_samples,impulse_position,upsample=upsample,draw_flag=draw_flag)
   passedFlag = TISC_sim(SNR,threshold,impulse_position,b_input_delay,c_input_delay,num_bits=num_bits,
                         upsample=upsample,num_samples=num_samples,noise_sigma=100,
                         cw_flag=cw_flag,peak_amplitude=cw_amplitude,carrier_frequency=cw_frequency,modulation_frequency=16000000.0,
                         draw_flag=draw_flag,digitization_factor=digitization_factor,output_dir="output/",delay_type_flag=delay_type_flag)
   print passedFlag
