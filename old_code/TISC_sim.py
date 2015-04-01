#!/usr/bin/env python

# A very simple impulsive signal + noise generator to get the 

import random
from ROOT import TTree, TBranch, gROOT, TStyle, TFile, gDirectory, TH1D, TCanvas, gStyle, TGraph
from array import array
import numpy as np
from impulse import impulse_gen

def sum_correlation(SNR,threshold,AinSignal,inBDelay,inCDelay,NoiseSigma=25,NoiseMean=380,fs=2400000000.0,LenTISCSample=16,LenEventWindow=512,seed=5522684,DrawCanvas=0):
   saveOutputFlag = 0
   gROOT.Reset()
   gStyle.SetOptStat("")
   gStyle.SetTitleFillColor(10)
   gStyle.SetTitleX(0.1)
   gStyle.SetTitleW(0.8)
   # Parameters
   #DrawCanvas = 1 # 0 don't draw the canvas, 1 draw it
   if(saveOutputFlag):
      outfile = "output/test.root"

   # Seed for random number generator
   #seed = 5522684 
   
   #LenEventWindow = 512 # Number of samples to generate
   #LenTISCSample = 16 # Number of samples GLITC can add at once
   #fs = 2400000000.0 # Sampling frequency
   Nbits = 3 # Number of bits available to the digitizer
   #NoiseMean = 380 # Mean for generating gaussian noise
   #NoiseSigma = 25 # Sigma for generating gaussian noise
   #ImpulsePos = 50 # 0 to LenEventWindow-LenTISCSample
   #inBDelay = 0 # Number of samples Ch B is delayed from A
   #inCDelay = 5 # Number of samples Ch C is delayed from A
   #SNR = 5 # Set the SNR of the event (Peak-to-peak of signal/RMS noise)
   #SignalAmp = NoiseSigma*SNR
   #antResponse = 1
   #chunkSize = 32
   triggerFlag = 0
   
   # Fill lists with zeros
   AinNoise = np.zeros(LenEventWindow)
   BinNoise = np.zeros(LenEventWindow)
   CinNoise = np.zeros(LenEventWindow)
   time = np.zeros(LenEventWindow)
   #AinSignal = np.zeros(LenEventWindow)
   BinSignal = np.zeros(LenEventWindow)
   CinSignal = np.zeros(LenEventWindow)
   AinSignalNoise = np.zeros(LenEventWindow)
   BinSignalNoise = np.zeros(LenEventWindow)
   CinSignalNoise = np.zeros(LenEventWindow)
   ANbitWaveform = np.zeros(LenEventWindow)
   BNbitWaveform = np.zeros(LenEventWindow)
   CNbitWaveform = np.zeros(LenEventWindow)
   EmptyList = np.zeros(LenEventWindow)

   if(saveOutputFlag):
      # Set up ROOT stuff
      f = TFile(outfile,"recreate")
      rf_tree = TTree("input_rf", "RF Tree")
      rf_tree.Branch('time',time,'time['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_A_Noise',AinNoise,'AinNoise['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_B_Noise',BinNoise,'BinNoise['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_C_Noise',CinNoise,'CinNoise['+str(LenEventWindow)+']/D')
      
      rf_tree.Branch('Ch_A_Signal',AinSignal,'AinSignal['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_B_Signal',BinSignal,'BinSignal['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_C_Signal',CinSignal,'CinSignal['+str(LenEventWindow)+']/D')
      
      rf_tree.Branch('Ch_A_SigNos',AinSignalNoise,'AinSignalNoise['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_B_SigNos',BinSignalNoise,'BinSignalNoise['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_C_SigNos',CinSignalNoise,'CinSignalNoise['+str(LenEventWindow)+']/D')
      
      rf_tree.Branch('Ch_A_3Bit',ANbitWaveform,'ANbitWaveform['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_B_3Bit',BNbitWaveform,'BNbitWaveform['+str(LenEventWindow)+']/D')
      rf_tree.Branch('Ch_C_3Bit',CNbitWaveform,'CNbitWaveform['+str(LenEventWindow)+']/D')
   
   
   
   # Generate the signal noise
   for i in range(0,LenEventWindow):
      AinNoise[i] = random.gauss(NoiseMean,NoiseSigma)
      BinNoise[i] = random.gauss(NoiseMean,NoiseSigma)
      CinNoise[i] = random.gauss(NoiseMean,NoiseSigma)
      time[i] = i*.357

   NoiseRMS = np.sqrt(np.mean((AinNoise-NoiseMean)**2,))
   #SignalAmp = np.sqrt(SNR)*NoiseRMS
   SignalAmp = SNR*NoiseRMS
   print "Noise RMS: " + str(NoiseRMS)
   print "Signal Amplitude: " + str(SignalAmp)
   print "SNR: " + str(SNR)
#################################

   # Generate Signal and Amplify
   #AinSignal[ImpulsePos] = SignalAmp # Generate delta function for testing
   #AinSignal = impulse_gen(LenEventWindow,ImpulsePos,SignalAmp,drawFlag=DrawCanvas)
   difference=np.amax(np.absolute(AinSignal))

   AinSignal *= (1/difference)
   AinSignal *= SignalAmp

   BinSignal = np.concatenate([EmptyList[:inBDelay],AinSignal[:LenEventWindow-inBDelay]])
   CinSignal = np.concatenate([EmptyList[:inCDelay],AinSignal[:LenEventWindow-inCDelay]])

   # Add the signal to the noise
   AinSignalNoise = np.add(AinNoise, AinSignal)
   BinSignalNoise = np.add(BinNoise, BinSignal)
   CinSignalNoise = np.add(CinNoise, CinSignal)
###################################

   # Convert to 3 bit values
   for i in range(0,LenEventWindow):
      for j in range(0,2**(Nbits)-3  ):
         if AinSignalNoise[i] > j*(NoiseRMS)+NoiseMean:
            ANbitWaveform[i] = j+Nbits+1
         elif AinSignalNoise[i] < (-j*(NoiseRMS)+NoiseMean):
            ANbitWaveform[i] = -j+Nbits+1

   for i in range(0,LenEventWindow):
      for j in range(0,2**(Nbits)-3  ):
         if BinSignalNoise[i] > j*(NoiseRMS)+NoiseMean:
            BNbitWaveform[i] = j+Nbits+1
         elif BinSignalNoise[i] < (-j*(NoiseRMS)+NoiseMean):
            BNbitWaveform[i] = -j+Nbits+1

   for i in range(0,LenEventWindow):
      for j in range(0,2**(Nbits)-3  ):
         if CinSignalNoise[i] > j*(NoiseRMS)+NoiseMean:
            CNbitWaveform[i] = j+Nbits+1
         elif CinSignalNoise[i] < (-j*(NoiseRMS)+NoiseMean):
            CNbitWaveform[i] = -j+Nbits+1

   # Zero Mean the Waveforms
   ANbitWaveform = ANbitWaveform-2**(Nbits-1)
   BNbitWaveform = BNbitWaveform-2**(Nbits-1)
   CNbitWaveform = CNbitWaveform-2**(Nbits-1)

########################################

###############################################
# Do Sum correlation
###############################################

   prevSum = 0
   thisSum = 0
   totalSum = 0
   maxSum = 0
   #correlation = [0] * 32
   addAB = [0] * LenTISCSample
   addABC = [0] * LenTISCSample
   bestaddABC = [0] * LenTISCSample
   bestBdelay = 0
   bestCdelay = 0
   bestchunk = 0
   bLowerLimit = 0
   cLowerLimit = 0
   triggerFlag = 0

   # Set the lower index limit of Ch B/C
   for chunk in range(0,(LenEventWindow/LenTISCSample)-1):
      if ((LenTISCSample*(chunk-2)) < 0):
         bLowerLimit = 0
         cLowerLimit = 0
      else:
         bLowerLimit = LenTISCSample*(chunk-2)
         cLowerLimit = LenTISCSample*(chunk-2)
   
      # Shift Ch B/C around
      for b in range(bLowerLimit,(LenTISCSample*(chunk+1))+1):
         for c in range(cLowerLimit,(LenTISCSample*(chunk+1))+1):
            thisSum = 0
	    #print "delays: "+str(chunk*LenTISCSample)+':'+str((chunk*LenTISCSample)+LenTISCSample-1)+'\t'+str(b)+':'+str(LenTISCSample+b-1)+'\t'+str(c)+':'+str(LenTISCSample+c-1)
            addAB = np.add(ANbitWaveform[chunk*LenTISCSample:(chunk*LenTISCSample)+LenTISCSample-1],BNbitWaveform[b:LenTISCSample+b-1])
            addABC = np.add(addAB,CNbitWaveform[c:LenTISCSample+c-1])
         
            # Now do the sum
            for i in range(0,LenTISCSample-1):
               thisSum = thisSum + addABC[i]**2
               totalSum = prevSum + thisSum
               if (totalSum > maxSum):
                  maxSum = totalSum
                  maxthisSum = thisSum
                  if (thisSum > prevSum):
                     #print addABC
                     #print str(thisSum)+'\t'+str(prevSum)
                     #print "Saving delays"
                     bestchunk = chunk*LenTISCSample
                     bestBdelay = b-bestchunk
                     bestCdelay = c-bestchunk
                     bestaddABC = addABC
            #print "maxSum: " + str(maxSum)
            #print str(chunk)+'\t'+str(bestBdelay)+'\t'+str(bestCdelay)
         
         #Close c loop
      #Close b loop
      prevSum=maxthisSum

   if(maxSum > threshold):
      triggerFlag = 1
      #print "Event passes"
   print "Threshold: " + str(threshold)
   print "MaxSum: " + str(maxSum)
   #print "A Chunk: " + str(bestchunk)
   #print "B Delay: " + str(bestBdelay)+'\t'+str(inBDelay-bestBdelay)
   #print "C Delay: " + str(bestCdelay)+'\t'+str(inCDelay-bestCdelay)
   #print "\n\n"

###############################################


   if(saveOutputFlag):
      # Now to more ROOT stuff
      rf_tree.Fill()
      f.Write()
      f.Close()

   if DrawCanvas:
      c1 = TCanvas('c1','Signal Simulation',800,600)
      #c2 = TCanvas('c2','Test',800,600)
      #c1.cd()
      c1.Divide(3,4,0,0)
      #c1.cd(1)
      #c2.cd()
      #testGraph = TGraph(LenEventWindow,array('d',time),array('d',CNbitWaveform))
      #testGraph.GetYaxis().SetTitle("Test [ADC Counts]")
      #testGraph.GetXaxis().SetTitle("Time Bins[ns]")
      #testGraph.SetTitle("Test Graph")
      #testGraph.Draw("AL")
      
      c1.cd(1)
      AinSignalGraph = TGraph(LenEventWindow,array('d',time),array('d',AinSignal))
      AinSignalGraph.GetYaxis().SetTitle("Ch A In Signal [mV]")
      AinSignalGraph.GetXaxis().SetTitle("Time Bins [ns]")
      AinSignalGraph.SetTitle('Ch A Signal')
   
      BinSignalGraph = TGraph(LenEventWindow,array('d',time),array('d',BinSignal))
      BinSignalGraph.GetYaxis().SetTitle("Ch B In Signal [mV]")
      BinSignalGraph.GetXaxis().SetTitle("Time Bins [ns]")
      BinSignalGraph.SetTitle('Ch B Signal')
      
      CinSignalGraph = TGraph(LenEventWindow,array('d',time),array('d',CinSignal))
      CinSignalGraph.GetYaxis().SetTitle("Ch C In Signal [mV]")
      CinSignalGraph.GetXaxis().SetTitle('Time Bins [ns]')
      CinSignalGraph.SetTitle('Ch C Signal')
   
      AinNoiseGraph = TGraph(LenEventWindow,array('d',time),array('d',AinNoise))
      AinNoiseGraph.GetYaxis().SetTitle("Ch A In Noise")
      AinNoiseGraph.GetXaxis().SetTitle("Time Bins [ns]")
      AinNoiseGraph.SetTitle('Ch A Noise [mV]')
   
      BinNoiseGraph = TGraph(LenEventWindow,array('d',time),array('d',BinNoise))
      BinNoiseGraph.GetYaxis().SetTitle("Ch B In Noise")
      BinNoiseGraph.GetXaxis().SetTitle("Time Bins [ns]")
      BinNoiseGraph.SetTitle('Ch B Noise [mV]')
   
      CinNoiseGraph = TGraph(LenEventWindow,array('d',time),array('d',CinNoise))
      CinNoiseGraph.GetYaxis().SetTitle("Ch C In Noise")
      CinNoiseGraph.GetXaxis().SetTitle("Time Bins [ns]")
      CinNoiseGraph.SetTitle('Ch C Noise [mV]')
   
   
      #c3 = TCanvas("c3",'Ch A Signal + Noise',900,700)
      #c1.cd(7)
      AinSignalNoiseGraph = TGraph(LenEventWindow,array('d',time),array('d',AinSignalNoise))
      AinSignalNoiseGraph.GetYaxis().SetTitle("Ch A S&N [mV]")
      AinSignalNoiseGraph.GetXaxis().SetTitle("Time Bins [ns]")
      AinSignalNoiseGraph.SetTitle('Ch A Signal + Noise')
      
      #c1.cd(8)
      BinSignalNoiseGraph = TGraph(LenEventWindow,array('d',time),array('d',BinSignalNoise))
      BinSignalNoiseGraph.GetYaxis().SetTitle("Ch B S&N [mV]")
      BinSignalNoiseGraph.GetXaxis().SetTitle("Time Bins [ns]")
      BinSignalNoiseGraph.SetTitle('Ch B Signal + Noise')
   
      #c1.cd(9)
      CinSignalNoiseGraph = TGraph(LenEventWindow,array('d',time),array('d',CinSignalNoise))
      CinSignalNoiseGraph.GetYaxis().SetTitle("Ch C S&N [mV]")
      CinSignalNoiseGraph.GetXaxis().SetTitle("Time Bins [ns]")
      CinSignalNoiseGraph.SetTitle('Ch C Signal + Noise')
   

      #c4 = TCanvas('c4','Ch A 3-bit Digitized Waveform',900,700)
      #c1.cd(10)
      ANbitWaveformGraph = TGraph(LenEventWindow,array('d',time),array('d',ANbitWaveform))
      ANbitWaveformGraph.GetYaxis().SetTitle("Ch A 3-bit wf [ADC counts]")
      ANbitWaveformGraph.GetXaxis().SetTitle("Time Bins [ns]")
      ANbitWaveformGraph.SetTitle("Ch A RITC Output")

      #c1.cd(11)
      BNbitWaveformGraph = TGraph(LenEventWindow,array('d',time),array('d',BNbitWaveform))
      BNbitWaveformGraph.GetYaxis().SetTitle("Ch B 3-bit wf [ADC counts]")
      BNbitWaveformGraph.GetXaxis().SetTitle("Time Bins [ns]")
      BNbitWaveformGraph.SetTitle("Ch B RITC Output")

      #c1.cd(12)
      CNbitWaveformGraph = TGraph(LenEventWindow,array('d',time),array('d',CNbitWaveform))
      CNbitWaveformGraph.GetYaxis().SetTitle("Ch C 3-bit wf [ADC counts]")
      CNbitWaveformGraph.GetXaxis().SetTitle("Time Bins [ns]")
      CNbitWaveformGraph.SetTitle("Ch C RITC Output")

      #c1.cd(13)
      sumABCWaveformGraph = TGraph(LenEventWindow,array('d',time),array('d',bestaddABC))
      sumABCWaveformGraph.GetYaxis().SetTitle("Summed Waveform [ADC counts]")
      sumABCWaveformGraph.GetXaxis().SetTitle("Time Bins [ns]")
      sumABCWaveformGraph.SetTitle("Summed Waveform")


      #c2.cd()
      c1.cd(1)
      AinSignalGraph.Draw("AL")
      #AinSignalGraph.SetMinimum(-200)
      #AinSignalGraph.SetMaximum(200)
      #AinSignalGraph.Write()
   
      c1.cd(2)
      BinSignalGraph.Draw("AL")
      #BinSignalGraph.SetMinimum(-200)
      #BinSignalGraph.SetMaximum(200)
      #BinSignalGraph.Write()

      c1.cd(3)
      CinSignalGraph.Draw("AL")
      #CinSignalGraph.SetMinimum(-200)
      #CinSignalGraph.SetMaximum(200)
      #CinSignalGraph.Write()
   
      ######################################
      c1.cd(4)
      AinNoiseGraph.Draw("AL")
      AinNoiseGraph.SetMinimum(200)
      AinNoiseGraph.SetMaximum(600)
      #AinNoiseGraph.Write()

      c1.cd(5)
      BinNoiseGraph.Draw("AL")
      BinNoiseGraph.SetMinimum(200)
      BinNoiseGraph.SetMaximum(600)
      #BinNoiseGraph.Write()

      c1.cd(6)
      CinNoiseGraph.Draw("AL")
      CinNoiseGraph.SetMinimum(200)
      CinNoiseGraph.SetMaximum(600)
      #CinNoiseGraph.Write()
      ####################################
      
      c1.cd(7)
      AinSignalNoiseGraph.Draw('AL')
      AinSignalNoiseGraph.SetMinimum(200)
      AinSignalNoiseGraph.SetMaximum(600)
      #AinSignalNoiseGraph.Write()
   
      c1.cd(8)
      BinSignalNoiseGraph.Draw('AL')
      BinSignalNoiseGraph.SetMinimum(200)
      BinSignalNoiseGraph.SetMaximum(600)
      #BinSignalNoiseGraph.Write()
   
      c1.cd(9)
      CinSignalNoiseGraph.Draw('AL')
      CinSignalNoiseGraph.SetMinimum(200)
      CinSignalNoiseGraph.SetMaximum(600)
      #CinSignalNoiseGraph.Write()
      ########################################

      c1.cd(10)
      ANbitWaveformGraph.Draw('AL')
      #ANbitWaveformGraph.Write()

      c1.cd(11)
      BNbitWaveformGraph.Draw('AL')
      #BNbitWaveformGraph.Write()

      c1.cd(12)
      CNbitWaveformGraph.Draw('AL')
      #CNbitWaveformGraph.Write()

      #c1.cd(13)
      #sumABCWaveformGraph.Draw('AL')
      c1.Update()
      dummy = raw_input('Press any key to close')

   return triggerFlag

if __name__ == '__main__':
   import ROOT
   import time
   LenEventWindow = 128
   ImpulsePos = 50
   SignalAmp = 10
   DrawFlag = 1
   BinDelay = 3
   CinDelay = 5
   lowSNR = 7
   highSNR = 8
   stepSNR = 1
   lowThreshold = 000
   highThreshold = 500
   stepThreshold = 10
   numRuns = 100

   # Get impulse signal
   AinSignal = np.zeros(LenEventWindow)
   AinSignal = impulse_gen(LenEventWindow,ImpulsePos,drawFlag=DrawFlag)
   SNR_hist = {}

   # Set up ROOT histogram
   SNRcanvas = ROOT.TCanvas('TISC_SNR', "TISC_SNR",200,10,700,500)
   ROOT.gROOT.Reset()
   ROOT.gStyle.SetOptStat("")
   ROOT.gStyle.SetTitleFillColor(10)
   ROOT.gStyle.SetTitleX(0.1)
   ROOT.gStyle.SetTitleW(0.8)

   outfile = TFile("output/TISC_SNR_"+str(time.strftime('%Y_%m_%d_%H.%m.%S'))+".root","RECREATE")
   #SNR_hist[1] = ROOT.TH1D('TISC_SNR','TISC SNR Curves',highThreshold/stepThreshold,lowThreshold,highThreshold)
   SNR_hist[0] = ROOT.TH1D('TISC_SNR','TISC SNR Curves',highThreshold/stepThreshold,lowThreshold,highThreshold)
   SNR_hist[0].SetMaximum(numRuns+1)
   SNR_hist[0].GetYaxis().SetTitle("Counts")
   SNR_hist[0].GetXaxis().SetTitle("Threshold")
   SNR_hist[0].Draw("*,C")
   x = np.linspace(lowThreshold,highThreshold,(highThreshold-lowThreshold)/stepThreshold)
   y = numRuns/2.0+0*x
   fiftyper = ROOT.TGraph(len(x),x,y)
   fiftyper.SetLineColor(20)
   fiftyper.Draw("SAME")

   legend = ROOT.TLegend(0.6,0.6,0.8,0.8)
   legend.AddEntry(fiftyper,"50% Efficency",'l')
   legend.Draw("SAME")

   #print "Exited impulse_gen"
   #print AinSignal
   for SNR in range(lowSNR,highSNR,stepSNR):
      SNR_hist[SNR] = ROOT.TH1D('TISC_SNR_'+str(SNR),'TISC SNR Curves',highThreshold/stepThreshold,lowThreshold,highThreshold)
      for threshold in range(lowThreshold,highThreshold,stepThreshold):
         for runNum in range(0,numRuns):
            passedFlag = 0
            passedFlag = sum_correlation(SNR,threshold,AinSignal,BinDelay,CinDelay,LenEventWindow=LenEventWindow,DrawCanvas=DrawFlag)
            if(passedFlag):
               SNR_hist[SNR].Fill(threshold)
               print "This event passed"
               print ">>>>>>>>>>>>>>>>>"
   
      SNR_hist[SNR].SetLineColor(SNR+1)
      SNR_hist[SNR].Draw("*,C,SAME")
      legend.AddEntry(SNR_hist[SNR],"SNR="+str(SNR),'l')
      legend.Draw("SAME")
   
   #img = ROOT.TImage.Create()
   #img.ROOT.FromPad(SNRcanvas)
   #img.WriteImage("SNR_Canvas."+str(time.strftime('%Y_%m_%d_%H.%m.%S'))+'.png')
   outfile.WriteTObject(SNRcanvas)

   outfile.Write()
   outfile.Close()
   outfile.ls()
   dummy = raw_input('Press any key to close')