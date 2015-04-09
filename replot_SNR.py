#!/usr/bin/env python

if __name__ == '__main__':
	from ROOT import TGraph, TMultiGraph,TCanvas,TImage,gROOT,gStyle,gSystem,gPad,TFile,TLegend, TH1F
	import numpy as np


	SNR_Graph = {}

	input_filename = 'test.root'
	f = TFile(input_filename)
	event_rate_graph = f.Get("Event Rate;1")
	max_trigger_rate_graph = f.Get("Maximum Trigger Rate;1")
	SNR_Graph[0] = f.Get("SNR_0.0;1")
	SNR_Graph[1] = f.Get("SNR_1.0;1")
	SNR_Graph[2] = f.Get("SNR_2.0;1")
	SNR_Graph[3] = f.Get("SNR_3.0;1")
	SNR_Graph[4] = f.Get("SNR_4.0:1")
	SNR_Graph[5] = f.Get("SNR_5.0;1")
	SNR_Graph[6] = f.Get("SNR_6.0;1")
	SNR_Graph[7] = f.Get("SNR_7.0;1")
	SNR_Graph[8] = f.Get("SNR_8.0;1")
	
	


	# Set up ROOT Stuff
	gROOT.Reset()
	gStyle.SetOptStat("")
	gStyle.SetTitleFillColor(10)
	gStyle.SetTitleX(0.1)
	gStyle.SetTitleW(0.8)

	
	SNR_canvas = TCanvas('TISC_SNR', "TISC_SNR",800,600)
	SNR_canvas.SetLogy()
	multigraph = TMultiGraph('SNR_Graph','SNR_Graph')
	
	max_trigger_rate_graph.SetLineColor(20)
	max_trigger_rate_graph.SetName("Maximum Trigger Rate")
	max_trigger_rate_graph.SetTitle("Maximum Trigger Rate")
	multigraph.Add(max_trigger_rate_graph)
	
	event_rate_graph.SetLineColor(46)
	event_rate_graph.SetName("Event Rate")
	event_rate_graph.SetTitle("Event Rate")
	multigraph.Add(event_rate_graph)
	
	line_color=1
	
	for SNR_counter in range(0,8):
		#SNR_Graph[SNR_counter].SetName('SNR '+str(SNR_counter))
		#SNR_Graph[SNR_counter].SetTitle('SNR '+str(SNR_counter))
		#SNR_Graph[SNR_counter].SetLineColor(line_color)
		#SNR_Graph[SNR_counter].SetLineWidth(2)
		#SNR_Graph[SNR_counter].SetFillStyle(0)
		#SNR_Graph[SNR_counter].SetDrawOption("A*")
		multigraph.Add(SNR_Graph[SNR_counter])
		line_color+=1
	
	
	#multigraph.SetTitle("TISC SNR Curves (@ "+str(event_rate/(10**6))+" MHz Event Rate); Threshold [~W]; Averaged Trigger Rate [Hz]")
	multigraph.Draw("ACP")
	SNR_canvas.BuildLegend(0.7,0.6,0.99,0.9)
	gPad.Update()
	gPad.Modified()
	
	dummy = raw_input('Press any key to close')
