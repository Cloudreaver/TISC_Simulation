#!/usr/bin/env python

if __name__ == '__main__':
	import time
	import os
	from avg_subtract_TISC_sim import avg_subtract_TISC_sim
	from TISC_sim import TISC_sim
	import numpy as np
	#import subprocess as Popen
	#from array import array
	import matplotlib.pyplot as plt
	
	
	#############################################################
	# Parameters
	num_events = 2		           		# Number of events to generate per threshold
	simulation_rate = 100.0   			# Simulation speed
	event_rate= 100.0			 		# Rate to generate impulseive events
	num_runs = int(num_events*(simulation_rate/event_rate))
	num_samples = 74                 	# Length of Signal Window
	noise_sigma = 20.0
	noise_mean = 0.0
	draw_flag = False                	# 1=draw event graphs, 0=don't                        
	SNR_draw_flag = True             	# 1=draw SNR graphs, 0=don't
	delay_type_flag = 1              	# 1=use GLITC delays, 0=all possible delays
	cw_flag = False
	carrier_frequency=260000000.0    	# Hz
	modulation_frequency=16000000.0  	# Hz
	peak_amplitude = 2.0*noise_sigma 	# Peak amplitude in mV
	save_output = True
	max_sum_memory = 4					# Number of max sums to average over
	

	impulse_pos = 5                  	# Position of impulse in Ch A (must be contained within first 32 samples)
	b_input_delay = 0                	# Ch B signal offset
	c_input_delay = 0                	# Ch C signal offset
	SNR = 3.0							# SNR of impulsive signal

	low_threshold = 0.0              	# Lowest Threshold
	high_threshold = 100.0           	# Highest Threshold
	step_threshold = 25.0            	# Threshold Interval
	#############################################################


	# Set some variables
	threshold = np.linspace(low_threshold, high_threshold,(high_threshold-low_threshold)/step_threshold+1)
	average_subtracted_trigger_rate = np.zeros(len(threshold))
	average_subtracted_trigger_number = np.zeros(len(threshold))
	trigger_rate = np.zeros(len(threshold))
	trigger_number = np.zeros(len(threshold))
	average_subtracted_noise_trigger_rate = np.zeros(len(threshold))
	average_subtracted_noise_trigger_number = np.zeros(len(threshold))
	noise_trigger_rate = np.zeros(len(threshold))
	noise_trigger_number = np.zeros(len(threshold))
	#average_signal=0
	average_subtracted_signal_max_sum=0
	#average_subtracted_noise_max_sum =0
	average_subtracted_signal_event_flag = 0
	average_subtracted_signal_noise_flag = 0
	mean_noise_max_sum = 0
	average_noise_max_sum = np.array([140,140,140,140])#np.zeros(max_sum_memory)
   
	# Write settings file
	if(save_output):
		if (cw_flag):
			output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/as_SNR"+str(SNR)+"_ER"+str(event_rate)+"_SR"+str(simulation_rate)+"_num"+str(num_events)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
		else:
			output_dir = str(os.path.dirname(os.path.realpath(__file__))+"/output/as_SNR"+str(SNR)+"_cw"+str(peak_amplitude/noise_sigma)+"_ER"+str(event_rate)+"_SR"+str(simulation_rate)+"_num"+str(num_events)+"_"+time.strftime('%Y_%m_%d_%H.%M.%S'))
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
	   
		settings_filename.write("\nCh B Input Delay: " + str(b_input_delay))
		settings_filename.write("\nCh C Input Delay: " + str(c_input_delay))
		settings_filename.write("\nNoise Mean: " +str(noise_mean))
		settings_filename.write("\nNoise Sigma: "+str(noise_sigma))
		settings_filename.write("\nMinimum Threshold: " + str(low_threshold))
		settings_filename.write("\nMaximum Threshold: " + str(high_threshold))
		settings_filename.write("\nThreshold Step: " + str(step_threshold))
		settings_filename.write("\nSimulation Rate: " + str(simulation_rate))
		settings_filename.write("\nImpulsive Event Rate: " + str(event_rate))

		if(save_output):
			data_filename = open(output_dir+"/output.dat","w")
   
	# Step over all desired thresholds
	for threshold_counter in range(0,len(threshold)):
		print "Starting threshold "+str(threshold[threshold_counter])
	   
		# Step over time.
		for timestep in range(0,num_runs):
			
			# Shift the average sum array to make room for new value
			average_noise_max_sum = np.roll(average_noise_max_sum,1)
				
			# Generate signal are event rate
			if(timestep % int((simulation_rate/event_rate)) ==0):
				#print "\nSending impulsive signal, timestep "+str(timestep)
				# Average subtracted impulsive event
				average_subtracted_event_passed_flag, average_subtracted_signal_max_sum = avg_subtract_TISC_sim(SNR,threshold[threshold_counter],
						impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,noise_mean=noise_mean,
						draw_flag=draw_flag,average_signal=mean_noise_max_sum)
				
				# Non subtracted impulseive event
				event_passed_flag, signal_max_sum = TISC_sim(SNR,threshold[threshold_counter],
						impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,noise_mean=noise_mean)
				
				# Average subtracted noise event
				average_subtracted_noise_passed_flag,average_noise_max_sum[0] = avg_subtract_TISC_sim(0.0,threshold[threshold_counter],
						impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,noise_mean=noise_mean,
						draw_flag=draw_flag,average_signal=mean_noise_max_sum)
						
				# non subtracted noise event					
				noise_passed_flag, noise_max_sum = TISC_sim(0.0,threshold[threshold_counter],
						impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,noise_mean=noise_mean)
							
			# Generate noise event when impulsive event is not called for
			else:
				#print "\nSending background signal, timestep "+str(timestep)
				# Average subtracted noise event
				average_subtracted_noise_passed_flag,average_noise_max_sum[0] = avg_subtract_TISC_sim(0,threshold[threshold_counter],
						impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,noise_mean=noise_mean,
						draw_flag=draw_flag,average_signal=mean_noise_max_sum)
						
				# Copy over values for subtracting on next iteration
				average_subtracted_event_passed_flag = average_subtracted_noise_passed_flag
						
				# Non subtracted noise event					
				noise_passed_flag, noise_max_sum = TISC_sim(0.0,threshold[threshold_counter],
						impulse_pos,b_input_delay,c_input_delay,num_samples=num_samples,
						cw_flag=cw_flag,carrier_frequency=carrier_frequency,peak_amplitude=peak_amplitude,modulation_frequency=modulation_frequency,
						noise_sigma=noise_sigma,noise_mean=noise_mean)
						
				# Copy over values for 
				event_passed_flag = noise_passed_flag
				event_max_sum = noise_max_sum		
			# Exit (event or noise) if statement
			
			# Take the mean of the max sum array
			mean_noise_max_sum = np.mean(average_noise_max_sum)
				
			# If trigger was seen, add it the trigger number					
			if(average_subtracted_event_passed_flag):
				average_subtracted_trigger_number[threshold_counter] +=1
			if(event_passed_flag):
				trigger_number[threshold_counter] +=1
			if(average_subtracted_noise_passed_flag):
				average_subtracted_noise_trigger_number[threshold_counter] +=1
			if(noise_passed_flag):
				noise_trigger_number[threshold_counter] +=1
				
			#print average_subtracted_trigger_number[threshold_counter]
			#print trigger_number[threshold_counter]
			#print average_subtracted_noise_trigger_number[threshold_counter]
			#print noise_trigger_number[threshold_counter]
			#print "\n"
		# Exit timestep loop
		
		if(save_output):
			data_filename.write(str(threshold[threshold_counter])+','+str(average_subtracted_trigger_number[threshold_counter])+','+str(trigger_number[threshold_counter])+','+str(average_subtracted_noise_trigger_number[threshold_counter])+','+str(noise_trigger_number[threshold_counter]))
	# Exit threshold counter loop	
		
	# Calculate trigger rates from number of triggers and the simulation rate
	average_subtracted_trigger_rate = (average_subtracted_trigger_number/num_runs)*simulation_rate
	trigger_rate = (trigger_number/num_runs)*simulation_rate
	average_subtracted_noise_trigger_rate = (average_subtracted_noise_trigger_number/num_runs)*simulation_rate
	noise_trigger_rate = (noise_trigger_number/num_runs)*simulation_rate
	

	
	# Make all the plots	
	plt.figure(1)
	plt.clf()
	plt.axis([np.amin(threshold),np.amax(threshold),0,(simulation_rate*1.5)])
	ax= plt.gca()
	ax.set_autoscale_on(False)
	plt.plot(threshold,average_subtracted_trigger_rate,'r',threshold,trigger_rate,'b',threshold,average_subtracted_noise_trigger_rate,'r--',threshold,noise_trigger_rate,'b--')
	plt.title("SNR Curves for "+str(event_rate)+" Hz Event Rate @ "+str(simulation_rate)+" Hz Simulation Rate")
	plt.xlabel("Threshold [DAC Counts]")
	plt.ylabel("Trigger Rate [Hz]")
	if (cw_flag):
		ax.legend((("Avg. Subtract SNR "+str(SNR)+" CW "+str(peak_amplitude/noise_sigma)),("Non-subtracted SNR "+str(SNR)+" CW "+str(peak_amplitude/noise_sigma)),"Avg. Subtracted Noise","Non-subtracted Noise"))
		if(save_output):
			plt.savefig(output_dir+"/avg_sub_cw_"+str(peak_amplitude/noise_sigma)+".png")	
	else:
		ax.legend((("Avg. Subtract SNR "+str(SNR)),("Non-Subtracted SNR "+str(SNR)),"Avg. Subtracted Noise","Non-Subtracted Noise"))
		if(save_output):
			plt.savefig(output_dir+"/avg_sub_vs_non_sub.png")
	plt.show()
	
	# Close open files
	if(save_output):
		settings_filename.close()
		data_filename.close()
           
