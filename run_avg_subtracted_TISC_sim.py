#!/usr/bin/env python

if __name__ == '__main__':
	import time
	from datetime import datetime
	import os
	from TISC_sim import TISC_sim
	import numpy as np
	#import subprocess as Popen
	#from array import array
	#import matplotlib
	#matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	##################### Parameters ###################################
	num_events = 1000  					# Number of events to generate per threshold
	simulation_rate = 200.0   			# Simulation speed
	event_rate= 100.0			 		# Rate to generate impulseive events
	num_trials = 100					# Number of trials to average over
	
	num_samples = 80                 	# Length of Signal Window (TISC needs at least 68 samples to calculate all delays
	upsample = 10						# Oversample factor for impulse generation
	
	# These delays should be negative, since A is the top antenna
	# and we expect an upgoing signal
	b_input_delay = 0                	# Ch B signal offset
	c_input_delay = 0                	# Ch C signal offset
	
	SNR = 3.0							# SNR of impulsive signal	
	noise_sigma = 20.0					# Gaussian Sigma in mV
	
	digitization_factor = 20.0       	# Digitization Factor
	
	draw_flag = False                	# 1=draw event graphs, 0=don't                        
	SNR_draw_flag = True             	# 1=draw SNR graphs, 0=don't
	delay_type_flag = True              # True=use GLITC delays, False=all possible delays
	cw_flag = True						# Turns on/off CW
	save_output = True					# To save or not to save
	
	carrier_frequency= 500000000.0    	# Hz
	modulation_frequency= 1.0  			# Hz
	peak_amplitude = 1.5*20.0 			# Peak amplitude in mV

	low_threshold = 0              		# Lowest Threshold
	high_threshold = 400          		# Highest Threshold
	step_threshold = 10            		# Threshold Interval
	
	max_sum_memory = 4					# Number of max sums to average over
	max_sum_offset = 50.0				# Shift the average subtracted plots back into view
	####################################################################

	################# Set some variables ###############################
	num_runs = int(num_events*(simulation_rate/event_rate))
	threshold = np.linspace(low_threshold, high_threshold,(high_threshold-low_threshold)/step_threshold+1)
	
	average_subtracted_trigger_rate = np.zeros(len(threshold))
	average_subtracted_trigger_number = np.zeros(len(threshold))
	trigger_rate = np.zeros(len(threshold))
	trigger_number = np.zeros(len(threshold))
	thermal_correlation_mean = np.zeros(50)
	#thermal_correlation_mean.fill(correlation_mean_primer)
	cw_thermal_correlation_mean = np.zeros(50)
	#cw_thermal_correlation_mean.fill(correlation_mean_primer)
	
	if(cw_flag):
		average_subtracted_cw_thermal_trigger_rate = np.zeros(len(threshold))
		average_subtracted_cw_thermal_trigger_number = np.zeros(len(threshold))
		cw_thermal_trigger_rate = np.zeros(len(threshold))
		cw_thermal_trigger_number = np.zeros(len(threshold))
	
	average_subtracted_thermal_trigger_rate = np.zeros(len(threshold))
	average_subtracted_thermal_trigger_number = np.zeros(len(threshold))
	thermal_trigger_rate = np.zeros(len(threshold))
	thermal_trigger_number = np.zeros(len(threshold))
	
	average_subtracted_cw_thermal_rate_diff = np.zeros(len(threshold))
	cw_thermal_rate_diff = np.zeros(len(threshold))
	
	fifty_percent = np.zeros(len(threshold))
	
	plot_correlation = np.zeros(num_trials)
	plot_time = np.linspace(0,num_trials,num_trials)
	#print len(plot_correlation)
	#print len(plot_time)
	
	#as_cw_thermal_max_sum = np.zeros(max_sum_memory)
	#as_cw_thermal_max_sum.fill(max_sum_memory_primer)
	#as_thermal_max_sum = np.zeros(max_sum_memory)
	#as_thermal_max_sum.fill(max_sum_memory_primer)
	

	####################################################################

	########### Write settings file ####################################
	if(save_output):
		if (cw_flag):
			output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/as_SNR"+str(SNR)+"_CW"+str(peak_amplitude/noise_sigma)+"_ER"+str(event_rate)+"_SR"+str(simulation_rate)+"_RUNS"+str(num_events)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
		else:
			output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/as_SNR"+str(SNR)+"_ER"+str(event_rate)+"_SR"+str(simulation_rate)+"_RUNS"+str(num_events)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
		print output_dir
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		settings_filename = open(output_dir+"/settings.txt","w")
		settings_filename.write("TISC simulation using new signal averaging system")
		settings_filename.write("Input parameters for TISC simulation\n\n")
		settings_filename.write("Start time: "+str(time.strftime('%Y_%m_%d_%H.%M.%S')))
		settings_filename.write("\nNumber of events per run: " + str(num_runs))
		settings_filename.write("\nNumber of samples per event: " + str(num_samples))
		settings_filename.write("\nImpulse Position: " + str(impulse_pos))
		settings_filename.write("\nUsing GLITC Delays")

		if(cw_flag):
			settings_filename.write("\nAdding CW Noise")
			settings_filename.write("\nCW Frequency: " +str(carrier_frequency))
			settings_filename.write("\nModulation Frequency: " + str(modulation_frequency))
			settings_filename.write("\nCW Amplitude: " + str(peak_amplitude))
		else:
			settings_filename.write("\nNo CW added")
	   
		settings_filename.write("\nCh B Input Delay: " + str(b_input_delay))
		settings_filename.write("\nCh C Input Delay: " + str(c_input_delay))
		settings_filename.write("\nNoise Sigma: "+str(noise_sigma))
		settings_filename.write("\nMinimum Threshold: " + str(low_threshold))
		settings_filename.write("\nMaximum Threshold: " + str(high_threshold))
		settings_filename.write("\nThreshold Step: " + str(step_threshold))
		settings_filename.write("\nSimulation Rate: " + str(simulation_rate))
		settings_filename.write("\nImpulsive Event Rate: " + str(event_rate))
		settings_filename.write("\nSNR: " +str(SNR))

		if(save_output):
			data_filename = open(output_dir+"/output.dat","w")
			
	####################################################################
	
	############ Run trials to get noise average #######################
	for timestep in range(1,num_trials+1):
		#if(timestep % 10 == 0):
			#print "\nStarting trial: "+str(timestep)
		
		# Thermal event					
		thermal_passed_flag, thermal_max_sum, average_subtracted_thermal_max_sum, thermal_correlation_mean = TISC_sim(0.0,100,
				impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
				cw_flag=0,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
				noise_sigma=noise_sigma,average_subtract_flag=1,correlation_mean=thermal_correlation_mean,
				trial_run_number=timestep,digitization_factor=digitization_factor)

		if(cw_flag):		
			# CW + thermal noise event				
			cw_thermal_passed_flag, cw_thermal_max_sum, average_subtracted_cw_thermal_max_sum, cw_thermal_correlation_mean = TISC_sim(0.0,100,
					impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
					cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
					noise_sigma=noise_sigma,average_subtract_flag=1,
					correlation_mean=cw_thermal_correlation_mean,trial_run_number=timestep,digitization_factor=digitization_factor)
					
		plot_correlation[timestep-1] = cw_thermal_correlation_mean[10]
		#print plot_correlation[timestep-1]			
	# Set offsets for plots				
	max_offset = 100#int(np.mean(thermal_correlation_mean))
	print cw_thermal_correlation_mean
	print thermal_correlation_mean
	####################################################################
	#plt.figure(3, figsize=(16,12))
	#plt.clf()
	#plt.axis(0,100,0,10)
	#ax = plt.gca()
	#ax.set_autoscale_on(False)
	#plt.grid(True)	
	#plt.plot(plot_time,plot_correlation,'b')
	#plt.title("Trigger Rate Improvement @ SNR: "+str(SNR)+"with CW: "+str(peak_amplitude/noise_sigma)+" "+str(event_rate)+" Hz Event Rate @ "+str(simulation_rate)+" Hz Simulation Rate")
	#plt.xlabel("Threshold [DAC Counts]")
	#plt.ylabel("Trigger Rate Difference [Hz]")
	#plt.ylim(0.0,simulation_rate*2)	
	#ax.legend(("Avg. Sub. Rate Diff @ SNR: "+str(SNR)),("Non Sub. Rate Diff @ SNR: "+str(SNR)))
	#plt.show()
	
	####################### Step over time #############################
	for timestep in range(0,num_runs):
		#if(timestep % 10 == 0):
			#print "\nStarting event: "+str(timestep)
		# Reset sums
		signal_max_sum = 0
		average_subtracted_signal_max_sum = 0
		thermal_max_sum = 0
		average_subtracted_signal_max_sum = 0
		cw_thermal_max_sum = 0
		average_subtracted_cw_thermal_max_sum = 0
		
		# Thermal event					
		thermal_passed_flag, thermal_max_sum, average_subtracted_thermal_max_sum, thermal_correlation_mean = TISC_sim(0.0,100,
				impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
				cw_flag=0,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
				noise_sigma=noise_sigma,average_subtract_flag=1,correlation_mean=thermal_correlation_mean,
				trial_run_number=0, digitization_factor=digitization_factor)
		
		# A bit of cheating here, if an impulsive or cw event is called for, these values will be overwritten
		cw_thermal_correlation_sum = thermal_correlation_mean
		signal_max_sum = thermal_max_sum
		average_subtracted_signal_max_sum = average_subtracted_thermal_max_sum
		
		if(cw_flag):		
			
			# CW + thermal noise event				
			cw_thermal_passed_flag, cw_thermal_max_sum, average_subtracted_cw_thermal_max_sum, cw_thermal_correlation_mean = TISC_sim(0.0,100,
					impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
					cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
					noise_sigma=noise_sigma,average_subtract_flag=1,
					correlation_mean=cw_thermal_correlation_mean,trial_run_number=0,digitization_factor=digitization_factor)
					
			# A bit of cheating here, if an impulsive or cw event is called for, these values will be overwritten
			signal_max_sum = cw_thermal_max_sum
			average_subtracted_signal_max_sum = average_subtracted_cw_thermal_max_sum
			
		# Generate signal at event rate
		if(timestep % int((simulation_rate/event_rate)) ==0):
			
			# Non-subtracted impulseive event
			event_passed_flag, signal_max_sum, average_subtracted_signal_max_sum, correlation_mean = TISC_sim(SNR,100,
						impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,upsample=upsample,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,average_subtract_flag=1,
						correlation_mean=cw_thermal_correlation_mean,trial_run_number=0,digitization_factor=digitization_factor)
		################################################################
		#print average_subtracted_signal_max_sum+max_sum_offset
		#print signal_max_sum
		#print "\n"
		## Done working with the waveforms, now compare to thresholds ##
		for threshold_counter in range(0,len(threshold)):		
			fifty_percent[threshold_counter] = (0.5*event_rate)
		
			# If trigger was seen, add it the trigger number		
			# Do the average subtracted signal check			
			if(((average_subtracted_signal_max_sum+max_sum_offset)>threshold[threshold_counter])):
				average_subtracted_trigger_number[threshold_counter] +=1
			
			# Do the non subtracted signal check
			if((signal_max_sum>threshold[threshold_counter])):
				trigger_number[threshold_counter] +=1
				
			if(cw_flag):
				# Do the average subtracted CW+Thermal check
				if(((average_subtracted_cw_thermal_max_sum+max_sum_offset )>threshold[threshold_counter])):
					average_subtracted_cw_thermal_trigger_number[threshold_counter] +=1
					
				# Do the non subtracted CW+Thermal check
				if(cw_thermal_max_sum>threshold[threshold_counter]):
					cw_thermal_trigger_number[threshold_counter] +=1
					
			# Do the average subtracted thermal check
			if((average_subtracted_thermal_max_sum+max_sum_offset)>threshold[threshold_counter]):
				average_subtracted_thermal_trigger_number[threshold_counter] +=1
				
			# Do the non subtracted thremal checks
			if(thermal_max_sum>threshold[threshold_counter]):
				thermal_trigger_number[threshold_counter] +=1
			
	### Exit timestep loop #############################################
	
	################### Save the data ##################################
	if(save_output):
		for threshold_counter in range(0,len(threshold)):
			if(cw_flag):
				data_filename.write(str(threshold[threshold_counter])+','+str(average_subtracted_trigger_number[threshold_counter])+','+str(trigger_number[threshold_counter])+','+str(average_subtracted_cw_thermal_trigger_number[threshold_counter])+','+str(cw_thermal_trigger_number[threshold_counter])+','+str(average_subtracted_thermal_trigger_number[threshold_counter])+','+str(thermal_trigger_number[threshold_counter])+"\n")
			else:
				data_filename.write(str(threshold[threshold_counter])+','+str(average_subtracted_trigger_number[threshold_counter])+','+str(trigger_number[threshold_counter])+','+str(average_subtracted_thermal_trigger_number[threshold_counter])+','+str(thermal_trigger_number[threshold_counter])+"\n")	
	####################################################################
		
	# Calculate trigger rates from number of triggers and the simulation rate
	average_subtracted_trigger_rate = (average_subtracted_trigger_number/num_runs)*simulation_rate
	trigger_rate = (trigger_number/num_runs)*simulation_rate
	
	if(cw_flag):
		average_subtracted_cw_thermal_trigger_rate = (average_subtracted_cw_thermal_trigger_number/num_runs)*simulation_rate
		cw_thermal_trigger_rate = (cw_thermal_trigger_number/num_runs)*simulation_rate
	average_subtracted_thermal_trigger_rate = (average_subtracted_thermal_trigger_number/num_runs)*simulation_rate
	thermal_trigger_rate = (thermal_trigger_number/num_runs)*simulation_rate
	
	if(cw_flag):
		average_subtracted_cw_thermal_rate_diff = np.subtract(average_subtracted_cw_thermal_trigger_rate,average_subtracted_thermal_trigger_rate)
		cw_thermal_rate_diff = np.subtract(cw_thermal_trigger_rate,thermal_trigger_rate)
	####################################################################

	################## Make all the plots ##############################	 
	plt.figure(1, figsize=(16, 12))
	plt.clf()
	plt.axis([np.amin(threshold),np.amax(threshold),event_rate/(10000.0),(simulation_rate*1.5)])
	ax= plt.gca()
	ax.set_autoscale_on(False)
	plt.grid(True)
	if(cw_flag):
		plt.semilogy(threshold,fifty_percent,'k',threshold,average_subtracted_trigger_rate,'r',threshold,trigger_rate,'b',threshold,average_subtracted_cw_thermal_trigger_rate,'r--',threshold,cw_thermal_trigger_rate,'b--',threshold,average_subtracted_thermal_trigger_rate,'r:',threshold,thermal_trigger_rate,'b:')
	else:
		plt.semilogy(threshold,fifty_percent,'k',threshold,average_subtracted_trigger_rate,'r',threshold,trigger_rate,'b',threshold,average_subtracted_thermal_trigger_rate,'r--',threshold,thermal_trigger_rate,'b--')
	plt.title("SNR Curves for "+str(event_rate)+" Hz Event Rate @ "+str(simulation_rate)+" Hz Simulation Rate")
	plt.xlabel("Threshold [DAC Counts]")
	plt.ylabel("Trigger Rate [Hz]")
	plt.ylim(0.0,simulation_rate*2)
	if (cw_flag):
		ax.legend((("50% Efficient"),("Avg. Subtract SNR "+str(SNR)+" CW "+str(peak_amplitude/noise_sigma)),("Non Subtracted SNR "+str(SNR)+" CW "+str(peak_amplitude/noise_sigma)),"Avg. Subtracted CW+Thermal","Non Subtracted CW+Thermal","Avg. Subtracted Thermal","Non Subtracted Thermal"))
		if(save_output):
			plt.savefig(output_dir+"/avg_sub_cw_"+str(peak_amplitude/noise_sigma)+".png")	
	else:
		ax.legend((("50% Efficient"),("Avg. Subtract SNR "+str(SNR)),("Non Subtracted SNR "+str(SNR)),"Avg. Subtracted Thermal","Non Subtracted Thermal"))
		if(save_output):
			plt.savefig(output_dir+"/avg_sub_vs_non_sub.png")
	
	# Plot differences between cw_thermal and thermal plots
	print average_subtracted_cw_thermal_trigger_rate
	average_subtracted_trigger_diff = np.subtract(average_subtracted_cw_thermal_trigger_rate,average_subtracted_thermal_trigger_rate)
	trigger_diff = np.subtract(cw_thermal_trigger_rate,thermal_trigger_rate)
	print average_subtracted_trigger_diff
	
	if(cw_flag):
		plt.figure(2, figsize=(16,12))
		plt.clf()
		plt.axis([np.amin(threshold),np.amax(threshold),event_rate/(10000.0),(simulation_rate*1.5)])
		ax= plt.gca()
		ax.set_autoscale_on(False)
		plt.grid(True)
		plt.semilogy(threshold,average_subtracted_trigger_diff,'r',threshold,trigger_diff,'b')
		plt.title("Trigger Rate Improvement @ SNR: "+str(SNR)+" with CW: "+str(peak_amplitude/noise_sigma)+" "+str(event_rate)+" Hz Event Rate @ "+str(simulation_rate)+" Hz Simulation Rate")
		plt.xlabel("Threshold [DAC Counts]")
		plt.ylabel("Trigger Rate Difference [Hz]")
		plt.ylim(0.0,simulation_rate*2)	
		#ax.legend(("Avg. Sub. Rate Diff @ SNR: "+str(SNR)),("Non Sub. Rate Diff @ SNR: "+str(SNR)))
		if(save_output):
			plt.savefig(output_dir+"/cw_vs_thermal_rates_"+str(peak_amplitude/noise_sigma)+".png")
	plt.show()
	
	# Close open files
	if(save_output):
		settings_filename.close()
		data_filename.close()
	
           
