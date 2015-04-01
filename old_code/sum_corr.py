#!/usr/bin/env python

# A very simple impulsive signal + noise generator to get the 

import random
from ROOT import TTree, TBranch, gROOT,gSystem,TImage, TStyle, TFile, gDirectory, TH1D, TCanvas, gStyle, TGraph
from array import array
import numpy as np
from impulse import impulse_gen, butter_bandpass_filter, butter_bandpass

def sum_correlation(SNR,threshold,a_input_signal,b_input_delay,c_input_delay,noise_sigma=25,noise_mean=380,sample_freq=2400000000.0,TISC_sample_length=16,num_samples=512,upsample=1,seed=5522684,draw_flag=0,digitization_factor=1,output_dir="output/"):
   gROOT.Reset()
   gStyle.SetOptStat("")
   gStyle.SetTitleFillColor(10)
   gStyle.SetTitleX(0.1)
   gStyle.SetTitleW(0.8)
   
   # Setup
   save_output_flag = 0
   if(save_output_flag):
      outfile = str(output_dir+"test.root")
   trigger_flag = 0
   num_bits = 3 # Number of bits available to the digitizer
   num_upsamples = num_samples*upsample
   
   # Fill numpy arrays with zeros
   a_input_noise = np.zeros(num_upsamples)
   b_input_noise = np.zeros(num_upsamples)
   c_input_noise = np.zeros(num_upsamples)
   time = np.zeros(num_samples)
   upsampled_time = np.zeros(num_upsamples)
   if SNR == 0:
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
   
   
   
   # Generate the signal noise
   for i in range(0,num_upsamples):
      a_input_noise[i] = random.gauss(0,noise_sigma)
      b_input_noise[i] = random.gauss(0,noise_sigma)
      c_input_noise[i] = random.gauss(0,noise_sigma)
      upsampled_time[i] = (i/(sample_freq*upsample))*(10**9)
   
   for i in range(0,num_samples):
      time[i] = (i/(sample_freq))*(10**9)
      
   if draw_flag:
      c5 = TCanvas('c5','Simulated Thermal Noise',1600,900)
      c5.cd()
      a_input_noise_graph1 = TGraph(num_upsamples,array('d',upsampled_time),array('d',a_input_noise))
      a_input_noise_graph1.GetYaxis().SetTitle("Voltage [V]")
      a_input_noise_graph1.GetXaxis().SetTitle("Time [ns]")
      a_input_noise_graph1.SetTitle('Thermal Noise')
      #a_input_noise_graph1.SetLineColor(1)
      #a_input_noise_graph1.GetHistogram().SetMaximum(600.)
      #a_input_noise_graph1.GetHistogram().SetMinimum(200.)
      a_input_noise_graph1.Draw()
      gSystem.ProcessEvents()
      img = TImage.Create()
      img.FromPad(c5)
      img.WriteImage(output_dir+"thermal_noise_no_filter.png")   

   # Bandlimit and upsample noise
   a_input_noise = (butter_bandpass_filter(a_input_noise)+noise_mean) # For some reason the filter zero means the noise
   b_input_noise = (butter_bandpass_filter(b_input_noise)+noise_mean)
   c_input_noise = (butter_bandpass_filter(c_input_noise)+noise_mean)

   if draw_flag:
      c6 = TCanvas('c6','Simulated Thermal Noise (Bandlimited)',1600,900)
      c6.cd()
      a_input_noise_graph2 = TGraph(num_upsamples,array('d',upsampled_time),array('d',a_input_noise))
      a_input_noise_graph2.GetYaxis().SetTitle("Voltage [V]")
      a_input_noise_graph2.GetXaxis().SetTitle("Time [ns]")
      a_input_noise_graph2.SetTitle('Thermal Noise (Bandlimited)')
      #a_input_noise_graph2.SetLineColor(2)
      #a_input_noise_graph2.GetHistogram().SetMaximum(600.)
      #a_input_noise_graph2.GetHistogram().SetMinimum(200.)
      a_input_noise_graph2.Draw()
      gSystem.ProcessEvents()
      img = TImage.Create()
      img.FromPad(c6)
      img.WriteImage(output_dir+"thermal_noise_with_filter.png")

   #print "Mean: " + str(np.mean((a_input_noise-noise_mean)))
   noise_rms = np.sqrt(np.mean((a_input_noise-noise_mean)**2,))
   #signal_amp = np.sqrt(SNR)*noise_rms
   signal_amp = SNR*noise_rms
   #print "Noise RMS: " + str(noise_rms)
   #print "Signal Amplitude: " + str(signal_amp)
   #print "SNR: " + str(SNR)
#################################
   if SNR != 0:
      # Generate Signal and Amplify
      #a_input_signal[impulse_position] = signal_amp # Generate delta function for testing
      #a_input_signal = impulse_gen(num_samples,impulse_position,signal_amp,draw_flag=draw_flag)
      difference=np.amax(np.absolute(a_input_signal))
      #print difference
      a_input_signal *= (1/difference)
      a_input_signal *= signal_amp
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
###################################
   noise_rms *= digitization_factor
   # Convert to 3 bit values
   for a_index in range(0,num_samples):
      for bit_index in range(0,2**(num_bits-1)):
         if (upsample > 1):
            downsample_random = random.randrange(0,upsample-1)
         else:
            downsample_random = 0
         if a_input_signal_noise[(a_index*upsample)+downsample_random] > (bit_index)*(noise_rms)+noise_mean: 
            a_dig_waveform[a_index] = bit_index+0.5
         elif a_input_signal_noise[(a_index*upsample)+downsample_random] < ((-bit_index)*(noise_rms)+noise_mean): 
            a_dig_waveform[a_index] = -bit_index-0.5
         

   for b_index in range(0,num_samples):
      for bit_index in range(0,2**(num_bits-1)):
         if (upsample > 1):
            downsample_random = random.randrange(0,upsample-1)
         else:
            downsample_random = 0
         if b_input_signal_noise[(b_index*upsample)+downsample_random] > (bit_index)*(noise_rms)+noise_mean:
            b_dig_waveform[b_index] = bit_index+0.5
         elif b_input_signal_noise[(b_index*upsample)+downsample_random] < ((-bit_index)*(noise_rms)+noise_mean):
            b_dig_waveform[b_index] = -bit_index-0.5

   for c_index in range(0,num_samples):
      for bit_index in range(0,2**(num_bits-1)):
         if (upsample > 1):
            downsample_random = random.randrange(0,upsample-1)
         else:
            downsample_random = 0
         if c_input_signal_noise[(c_index*upsample)+downsample_random] > (bit_index)*(noise_rms)+noise_mean:
            c_dig_waveform[c_index] = bit_index+0.5
         elif c_input_signal_noise[(c_index*upsample)+downsample_random] < ((-bit_index)*(noise_rms)+noise_mean):
            c_dig_waveform[c_index] = -bit_index-0.5

   # Zero Mean the Waveforms
   #a_dig_waveform = a_dig_waveform-2**(num_bits-1)
   #b_dig_waveform = b_dig_waveform-2**(num_bits-1)
   #c_dig_waveform = c_dig_waveform-2**(num_bits-1)

########################################

###############################################
# Do Sum correlation
###############################################

   prev_sum = 0
   this_sum = 0
   total_sum = 0
   max_total_sum = 0
   #correlation = [0] * 32
   add_AB = [0] * TISC_sample_length
   add_ABC = [0] * TISC_sample_length
   best_add_ABC = [0] * TISC_sample_length
   best_b_delay = 0
   best_c_delay = 0
   best_chunk = 0
   b_lower_index_limit = 0
   c_lower_index_limit = 0
   trigger_flag = 0

   # Set the lower index limit of Ch B/C
   for chunk in range(0,(num_samples/TISC_sample_length)-1):
      if ((TISC_sample_length*(chunk-2)) < 0):
         b_lower_index_limit = 0
         c_lower_index_limit = 0
      else:
         b_lower_index_limit = TISC_sample_length*(chunk-2)
         c_lower_index_limit = TISC_sample_length*(chunk-2)
   
      # Shift Ch B/C around
      for b in range(b_lower_index_limit,(TISC_sample_length*(chunk+1))+1):
         for c in range(c_lower_index_limit,(TISC_sample_length*(chunk+1))+1):
            this_sum = 0
	    #print "delays: "+str(chunk*TISC_sample_length)+':'+str((chunk*TISC_sample_length)+TISC_sample_length-1)+'\t'+str(b)+':'+str(TISC_sample_length+b-1)+'\t'+str(c)+':'+str(TISC_sample_length+c-1)
            add_AB = np.add(a_dig_waveform[chunk*TISC_sample_length:(chunk*TISC_sample_length)+TISC_sample_length-1],b_dig_waveform[b:TISC_sample_length+b-1])
            add_ABC = np.add(add_AB,c_dig_waveform[c:TISC_sample_length+c-1])
         
            # Now do the sum
            for i in range(0,TISC_sample_length-1):
               this_sum = this_sum + add_ABC[i]**2
               total_sum = prev_sum + this_sum
               if (total_sum > max_total_sum):
                  max_total_sum = total_sum
                  max_this_Sum = this_sum
                  if (this_sum > prev_sum):
                     #print add_ABC
                     #print str(this_sum)+'\t'+str(prev_sum)
                     #print "Saving delays"
                     best_chunk = chunk*TISC_sample_length
                     best_b_delay = b-best_chunk
                     best_c_delay = c-best_chunk
                     best_add_ABC = add_ABC
            #print "max_total_sum: " + str(max_total_sum)
            #print str(chunk)+'\t'+str(best_b_delay)+'\t'+str(best_c_delay)
         
         #Close c loop
      #Close b loop
      prev_sum=max_this_Sum

   if(max_total_sum > threshold):
      trigger_flag = 1
      #print "Event passes"
   #print "Threshold: " + str(threshold)
   #print "max_total_sum: " + str(max_total_sum)
   #print "A Chunk: " + str(best_chunk)
   #print "B Delay: " + str(best_b_delay)+'\t'+str(b_input_delay-best_b_delay)
   #print "C Delay: " + str(best_c_delay)+'\t'+str(c_input_delay-best_c_delay)
   #print "\n\n"

###############################################


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
      abc_sum_waveform_graph = TGraph(num_samples,array('d',time),array('d',best_add_ABC))
      abc_sum_waveform_graph.GetYaxis().SetTitle("Summed Waveform [ADC counts]")
      abc_sum_waveform_graph.GetXaxis().SetTitle("Time Bins [ns]")
      abc_sum_waveform_graph.SetTitle("Summed Waveform")


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
      a_input_noise_graph.SetMinimum(200)
      a_input_noise_graph.SetMaximum(600)
      #a_input_noise_graph.Write()

      c1.cd(5)
      b_input_noise_graph.Draw("AL")
      b_input_noise_graph.SetMinimum(200)
      b_input_noise_graph.SetMaximum(600)
      #b_input_noise_graph.Write()

      c1.cd(6)
      c_input_noise_graph.Draw("AL")
      c_input_noise_graph.SetMinimum(200)
      c_input_noise_graph.SetMaximum(600)
      #c_input_noise_graph.Write()
      ####################################
      
      c1.cd(7)
      a_input_signal_noise_graph.Draw('AL')
      a_input_signal_noise_graph.SetMinimum(200)
      a_input_signal_noise_graph.SetMaximum(600)
      #a_input_signal_noise_graph.Write()
   
      c1.cd(8)
      b_input_signal_noise_graph.Draw('AL')
      b_input_signal_noise_graph.SetMinimum(200)
      b_input_signal_noise_graph.SetMaximum(600)
      #b_input_signal_noise_graph.Write()
   
      c1.cd(9)
      c_input_signal_noise_graph.Draw('AL')
      c_input_signal_noise_graph.SetMinimum(200)
      c_input_signal_noise_graph.SetMaximum(600)
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
      dummy = raw_input('Press any key to close')

   return trigger_flag

if __name__ == '__main__':
   import ROOT
   import time
   num_samples = 48
   impulse_position = 10
   signal_amp = 10
   draw_flag = 1
   b_input_delay = 3
   c_input_delay = 5
   SNR = 5
   Threshold = 300
   #num_runs = 100
   upsample = 2

   # Get impulse signal
   a_input_signal = np.zeros(num_samples*upsample)
   a_input_signal = impulse_gen(num_samples,impulse_position,upsample=upsample,draw_flag=draw_flag)
   passedFlag = sum_correlation(SNR,Threshold,a_input_signal,b_input_delay,c_input_delay,upsample=upsample,num_samples=num_samples,draw_flag=draw_flag,digitization_factor=1,output_dir="output/")
