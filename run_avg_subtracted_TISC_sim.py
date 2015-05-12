#!/usr/bin/env python

if __name__ == '__main__':
	import time
	from datetime import datetime
	import os
	from TISC_sim import TISC_sim
	import numpy as np
	#import subprocess as Popen
	#from array import array
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	start_time = datetime.now()
	#############################################################
	# Parameters
	num_events = 100000       			# Number of events to generate per threshold
	simulation_rate = 200.0   			# Simulation speed
	event_rate= 100.0			 		# Rate to generate impulseive events
	num_runs = int(num_events*(simulation_rate/event_rate))
	num_samples = 74                 	# Length of Signal Window
	noise_sigma = 20.0
	noise_mean = 0.0
	draw_flag = False                	# 1=draw event graphs, 0=don't                        
	SNR_draw_flag = True             	# 1=draw SNR graphs, 0=don't
	delay_type_flag = 1              	# 1=use GLITC delays, 0=all possible delays
	cw_flag = True
	carrier_frequency=260000000.0    	# Hz
	modulation_frequency=1.0  			# Hz
	peak_amplitude = 2.0*noise_sigma 	# Peak amplitude in mV
	save_output = True					# To save or not to save
	max_sum_memory = 4					# Number of max sums to average over
	

	impulse_pos = 5                  	# Position of impulse in Ch A (must be contained within first 32 samples)
	b_input_delay = 0                	# Ch B signal offset
	c_input_delay = 0                	# Ch C signal offset
	SNR = 3.0							# SNR of impulsive signal

	low_threshold = 0              		# Lowest Threshold
	high_threshold = 1000          		# Highest Threshold
	step_threshold = 1            		# Threshold Interval
	
	max_sum_offset = 150				# Shift the average subtracted plots back into view
	#############################################################

	# Set some variables
	threshold = np.linspace(low_threshold, high_threshold,(high_threshold-low_threshold)/step_threshold+1)
	average_subtracted_trigger_rate = np.zeros(len(threshold))
	average_subtracted_trigger_number = np.zeros(len(threshold))
	trigger_rate = np.zeros(len(threshold))
	trigger_number = np.zeros(len(threshold))
	
	if(cw_flag):
		average_subtracted_cw_thermal_trigger_rate = np.zeros(len(threshold))
		average_subtracted_cw_thermal_trigger_number = np.zeros(len(threshold))
		cw_thermal_trigger_rate = np.zeros(len(threshold))
		cw_thermal_trigger_number = np.zeros(len(threshold))
	
	average_subtracted_thermal_trigger_rate = np.zeros(len(threshold))
	average_subtracted_thermal_trigger_number = np.zeros(len(threshold))
	thermal_trigger_rate = np.zeros(len(threshold))
	thermal_trigger_number = np.zeros(len(threshold))
	
	fifty_percent = np.zeros(len(threshold))

	# Write settings file
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
		settings_filename.write("\nNoise Mean: " +str(noise_mean))
		settings_filename.write("\nNoise Sigma: "+str(noise_sigma))
		settings_filename.write("\nMinimum Threshold: " + str(low_threshold))
		settings_filename.write("\nMaximum Threshold: " + str(high_threshold))
		settings_filename.write("\nThreshold Step: " + str(step_threshold))
		settings_filename.write("\nSimulation Rate: " + str(simulation_rate))
		settings_filename.write("\nImpulsive Event Rate: " + str(event_rate))
		settings_filename.write("\nSNR: " +str(SNR))

		if(save_output):
			data_filename = open(output_dir+"/output.dat","w")
			
	as_cw_thermal_max_sum = np.array([140,140,140,140])
	as_thermal_max_sum = np.array([140,140,140,140])
		
	# Step over time.
	for timestep in range(0,num_runs):
		print "\nStarting timestep: "+str(timestep)
		average_subtracted_signal_max_sum = 0
		thermal_max_sum = 0
		cw_thermal_max_sum = 0
		signal_max_sum = 0

		# Shift the average arrays
		as_cw_thermal_max_sum = np.roll(as_cw_thermal_max_sum,1)
		as_thermal_max_sum = np.roll(as_thermal_max_sum,1)
		
		# Take the mean of the max sum array
		mean_cw_thermal_max_sum = np.mean(as_cw_thermal_max_sum)
		mean_thermal_max_sum = np.mean(as_thermal_max_sum)
	
		# Thermal event					
		thermal_passed_flag, thermal_max_sum = TISC_sim(0.0,100,
				impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
				cw_flag=0,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
				noise_sigma=noise_sigma,noise_mean=noise_mean)
				
		# A bit of cheating here, if an impulsive or cw event is called for, these values will be overwritten
		as_thermal_max_sum[0] = thermal_max_sum
		average_subtracted_signal_max_sum = as_thermal_max_sum[0]
		signal_max_sum = thermal_max_sum

		if(cw_flag):		
			# Non-subtracted CW + thermal noise event					
			cw_thermal_passed_flag, cw_thermal_max_sum = TISC_sim(0.0,100,
					impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
					cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
					noise_sigma=noise_sigma,noise_mean=noise_mean)
			as_cw_thermal_max_sum[0] = cw_thermal_max_sum

			# A bit of cheating here, if an impulsive event is called for, these values will be overwritten
			average_subtracted_signal_max_sum = as_cw_thermal_max_sum[0]
			signal_max_sum = cw_thermal_max_sum

		# Generate signal at event rate
		if(timestep % int((simulation_rate/event_rate)) ==0):
			#print "Sending impulsive signal, timestep "+str(timestep)
				
			# Non-subtracted impulseive event
			event_passed_flag, signal_max_sum = TISC_sim(SNR,100,
						impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,noise_mean=noise_mean)
			average_subtracted_signal_max_sum = signal_max_sum

		# Compare the outputs to the thresholds
		for threshold_counter in range(0,len(threshold)):
			#print "Starting threshold "+str(threshold[threshold_counter])
			
			fifty_percent[threshold_counter] = (0.5*event_rate)
		
			# If trigger was seen, add it the trigger number		
			# Do the average subtracted signal check			
			if(((average_subtracted_signal_max_sum - mean_cw_thermal_max_sum+max_sum_offset)>threshold[threshold_counter])):
				average_subtracted_trigger_number[threshold_counter] +=1
			
			# Do the non subtracted signal check
			if((signal_max_sum>threshold[threshold_counter])):
				trigger_number[threshold_counter] +=1
			if(cw_flag):
				# Do the average subtracted CW+Thermal check
				if(((as_cw_thermal_max_sum[0] - mean_cw_thermal_max_sum + max_sum_offset)>threshold[threshold_counter])):
					average_subtracted_cw_thermal_trigger_number[threshold_counter] +=1
				# Do the non subtracted CW+Thermal check
				if(cw_thermal_max_sum>threshold[threshold_counter]):
					cw_thermal_trigger_number[threshold_counter] +=1
			# Do the average subtracted thermal check
			if((as_thermal_max_sum[0] - mean_thermal_max_sum + max_sum_offset)>threshold[threshold_counter]):
				average_subtracted_thermal_trigger_number[threshold_counter] +=1
			# Do the non subtracted thremal checks
			if(thermal_max_sum>threshold[threshold_counter]):
				thermal_trigger_number[threshold_counter] +=1
			
	# Exit timestep loop
	
	# Save the data
	if(save_output):
		for threshold_counter in range(0,len(threshold)):
			if(cw_flag):
				data_filename.write(str(threshold[threshold_counter])+','+str(average_subtracted_trigger_number[threshold_counter])+','+str(trigger_number[threshold_counter])+','+str(average_subtracted_cw_thermal_trigger_number[threshold_counter])+','+str(cw_thermal_trigger_number[threshold_counter])+','+str(average_subtracted_thermal_trigger_number[threshold_counter])+','+str(thermal_trigger_number[threshold_counter])+"\n")
			else:
				data_filename.write(str(threshold[threshold_counter])+','+str(average_subtracted_trigger_number[threshold_counter])+','+str(trigger_number[threshold_counter])+','+str(average_subtracted_thermal_trigger_number[threshold_counter])+','+str(thermal_trigger_number[threshold_counter])+"\n")	
		
	# Calculate trigger rates from number of triggers and the simulation rate
	average_subtracted_trigger_rate = (average_subtracted_trigger_number/num_runs)*simulation_rate
	trigger_rate = (trigger_number/num_runs)*simulation_rate
	if(cw_flag):
		average_subtracted_cw_thermal_trigger_rate = (average_subtracted_cw_thermal_trigger_number/num_runs)*simulation_rate
		cw_thermal_trigger_rate = (cw_thermal_trigger_number/num_runs)*simulation_rate
	average_subtracted_thermal_trigger_rate = (average_subtracted_thermal_trigger_number/num_runs)*simulation_rate
	thermal_trigger_rate = (thermal_trigger_number/num_runs)*simulation_rate

	# Make all the plots	
	plt.figure(num=None, figsize=(16, 12))
	plt.clf()
	plt.axis([np.amin(threshold),np.amax(threshold),0,(simulation_rate*1.5)])
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
	#print datetime.now() - start_time
	#plt.show()
	
	# Close open files
	if(save_output):
		settings_filename.close()
		data_filename.close()
	
           
