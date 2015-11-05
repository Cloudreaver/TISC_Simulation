#!/usr/bin/env python

import numpy as np
import csv
import scipy

# Read the individual sample offsets file and returns a 2d matrix [sample]x[threshold_level]
def get_individual_sample_offsets(GLITC_n,channel):
	
	offsets = [[0 for x in range(8)] for x in range(32)]
	row_counter = 0
	with open('/home/hupe.2/thesis_code/testbench_analysis/G%d_Ch%d_sample_threshold_offsets.dat'%(GLITC_n,channel),'rb') as datafile:
		datareader = csv.reader(datafile, delimiter=',', quotechar='|')
		# Read in and plot the data
		for row in datareader:
			for thres_i in range(8):
				offsets[row_counter][thres_i] = float(row[thres_i])			
			row_counter+=1
	
	
	return offsets
	
	
# Take in a sample array and a offsets matrix and corrects it
def correct_individual_samples(sample_array,offsets):
	
	if(len(sample_array)%32!=0):
		print "Samples array must include all 32 samples and begin with sample 0!"
		return 1
	
	num_sample_blocks = int(len(sample_array)/32)
	
	return_array = np.zeros(len(sample_array))
	
	for block_i in range(num_sample_blocks):
		for samp_i in range(32):
			threshold_level = int(sample_array[32*block_i+samp_i])
			#print threshold_level
			return_array[32*block_i+samp_i] = sample_array[32*block_i+samp_i] + offsets[samp_i][threshold_level]
	
	return return_array
	
def plot_impulse_response():
	import matplotlib.pyplot as plt
	import scipy.stats
	
	plot_start = 40
	
	t1 = []
	a1 = []
	anita_power = []
	
	with open("/home/hupe.2/thesis_code/anita_II_impulse_response.dat",'rb') as csvfile:
		reader = csv.reader(csvfile,delimiter=' ', quotechar='|')
		for row in reader:
			t1.append(float(row[0]))
			a1.append(float(row[1]))
			
		
	peak_index = np.argmin(a1)
	
	t1 = np.array(t1)-t1[0]
	delta_t1 = t1[1]-t1[0]
	print delta_t1
	a1 = np.roll(a1,60-peak_index)
	length_1 = len(t1)
	print length_1
	print delta_t1*length_1
	a1 = np.array(a1)*(1.0/(np.amax(a1)-np.amin(a1)))
	peak_loc_1 = np.argmax(a1)
	#print np.amin(a1)
	#print np.amax(a1)
	plt.figure(1,figsize=(16,9))
	#plt.plot(t[plot_start:790+plot_start],a1[plot_start:790+plot_start],label="ANITA Impulse Response")
	plt.plot(t1*10**9,a1,label="ANITA Impulse Response")
	#plt.ylabel("Voltage (Normalized)")
	#plt.xlabel("Time [ns]")
	#plt.xlim(0,40)
	#plt.legend()
	#plt.grid(True)
	#plt.show()
	
	power_temp = 0
	for i in range(len(a1)):
		power_temp += a1[i]**2
		anita_power.append(power_temp)
	anita_power = np.array(anita_power)*(1.0/np.amax(anita_power))
	
	t2 = []
	a2 = []
	testbench_power = []
	
	with open("/home/hupe.2/thesis_code/testbench_analysis/data/testbench_impulse.dat",'rb') as csvfile:
		reader = csv.reader(csvfile,delimiter=',', quotechar='|')
		for row in reader:
			t2.append(float(row[0]))
			a2.append(float(row[1]))
	
	peak_index = np.argmin(a2)
	
	t2 = np.array(t2)-t2[0]
	#t = np.array(t)-0.0000000589
	delta_t2 = t2[1]-t2[0]
	print delta_t2
	a2 = np.roll(a2,int(60*(delta_t1/delta_t2))-peak_index-28)
	length_2 = len(t2)
	print length_2
	print delta_t2*length_2
	peak_loc_2 = np.argmax(a2)
	plot_samples = length_2#int((delta_t1*length_1)/delta_t2)
	plot_start=0
	#print len(t)
	a2 = (-1)*np.array(a2)*(1.0/(np.amax(a2)-np.amin(a2)))
	
	#plt.plot(t[plot_start:plot_samples+plot_start],np.array(a[plot_start:plot_samples+plot_start]),label="Testbench Impulse Response")
	plt.plot(t2*10**9,a2,label="Testbench Impulse Response")
	plt.legend()
	plt.title("ANITA Impulse Response vs Testbench Impulse Response")
	plt.ylabel("Voltage (Normalized)")
	plt.xlabel("Time [ns]")
	plt.xlim(0,60)
	plt.legend()
	plt.grid(True)
	#plt.show()
	anita_power = []
	anita_max_power = []
	num_anita_samples_in_window = int(12.4/(delta_t1*10**9))*2/2
	num_anita_sub_samples_in_window = num_anita_samples_in_window/2
	print len(a1)-num_anita_samples_in_window
	print num_anita_samples_in_window
	for i in range(0,len(a1)-num_anita_samples_in_window):
		#start1 =i
		#end1 = start1+num_anita_sub_samples_in_window
		#start2 = end1+1
		#end2 = start2+num_anita_sub_samples_in_window
		#print i
		#print start1,end1
		#print start2,end2
		#temp1 = np.mean(a1[start1:end1]**2)
		#temp2 = np.mean(a1[start2:end2]**2)
		#anita_max_power.append(temp1+temp2)
		anita_power.append(np.mean(a1[i:i+num_anita_samples_in_window]**2))
	for i in range(0,len(a1)-num_anita_sub_samples_in_window):
		anita_max_power.append(np.mean(anita_power[i:i+num_anita_sub_samples_in_window]))
		
		
	tb_power = []
	tb_max_power = []
	num_tb_samples_in_window = int(12.4/(delta_t2*10**9))
	num_tb_sub_samples_in_window = num_tb_samples_in_window/2
	#print len(a2)
	#print num_tb_samples_in_window
	#print num_tb_sub_samples_in_window
	#print len(a2)-num_tb_samples_in_window
	#print len(a2-num_tb_samples_in_window)
	for i in range(0,len(a2)-num_tb_samples_in_window):
		#start1 =i
		#end1 = start1+num_tb_sub_samples_in_window
		#start2 = end1+1
		#end2 = start2+num_tb_sub_samples_in_window
		#print i
		#print start1,end1
		#print start2,end2
		#temp1 = np.mean(a2[start1:end1]**2)
		#temp2 = np.mean(a2[start2:end2]**2)
		#tb_max_power.append(temp1+temp2)
		tb_power.append(np.mean(a2[i:i+num_tb_samples_in_window]**2))
	for i in range(0,len(a2)-num_tb_sub_samples_in_window):
		tb_max_power.append(np.mean(tb_power[i:i+num_tb_sub_samples_in_window]))
		
	plt.figure(2,figsize=(16,9))
	print len(t1),len(t2)
	print len(anita_power),len(tb_power)
	
	plt.plot(t1[:len(a1)-num_anita_samples_in_window]*10**9,anita_power,label='ANITA Impulse')
	plt.plot(t2[:len(a2)-num_tb_samples_in_window]*10**9,tb_power,label='Testbench Impulse')
	plt.title("32 Sample (12.4ns) Moving Average Power")
	plt.ylabel("Average Power")
	plt.xlabel("Time [ns]")
	plt.xlim(0,60)
	plt.legend()
	print np.amax(anita_power)
	print np.amax(tb_power)
	print np.sqrt(np.amax(anita_power)/np.amax(tb_power))
	
	plt.figure(3,figsize=(16,8))
	print len(t1),len(t2)
	print len(anita_power),len(tb_power)
	
	plt.plot(t1[:len(a1)-num_anita_sub_samples_in_window]*10**9,anita_max_power,label='ANITA Impulse')
	plt.plot(t2[:len(a2)-num_tb_sub_samples_in_window]*10**9,tb_max_power,label='Testbench Impulse')
	plt.title("16 Sample (6.2ns) Moving Average of 32 Sample (12.4ns) Moving Average Power")
	plt.ylabel("Average Power")
	plt.xlabel("Time [ns]")
	plt.xlim(0,60)
	plt.legend()
	
	
	"""
	power_temp = 0
	for i in range(len(a2)):
		power_temp += a2[i]**2
		testbench_power.append(power_temp) 
	
	testbench_power = np.array(testbench_power)*(1.0/np.amax(testbench_power))
	plt.figure(2)
	plt.plot(t1*10**9,np.array(anita_power)*(1.0/np.amax(anita_power)),label="ANITA Impulse Integrated Power")
	plt.plot(t2*10**9,np.array(testbench_power)*(1.0/np.amax(testbench_power)),label="Testbench Impulse Integrated Power")
	plt.title("ANITA Impulse Integrated Power vs Testbench Impulse Integrated Power")
	plt.ylabel("Integrated Power (Normilized)")
	plt.xlabel("Time [ns]")  
	plt.xlim(0,40)
	plt.legend(loc=0)
	plt.grid(True)
	
	anita_slope = np.diff(anita_power)
	testbench_slope = np.diff(testbench_power)
	anita_mu, anita_std = scipy.stats.norm.fit(anita_slope)
	tb_mu,tb_std = scipy.stats.norm.fit(testbench_slope)
	
	print t1[anita_mu]*10**9, t1[anita_std]*10**9
	print t2[tb_mu]*10**9], t2[tb_std]*10**9
	print (anita_mu)/(tb_mu)
	plt.figure(3)
	plt.plot(t1[:len(t1)-1]*10**9,anita_slope,label="ANITA Impulse Integrated Power Derivative")
	plt.plot(t2[:len(t2)-1]*10**9,testbench_slope,label="Testbench Impulse Integrated Power Derivative")
	plt.axvline(anita_mu*delta_t1*10**9)
	plt.axvline(tb_mu*delta_t2*10**9)
	plt.title("ANITA Impulse vs Testbench Impulse")
	plt.ylabel("Integrated Power (Normilized)")
	plt.xlabel("Time [ns]")
	plt.xlim(0,40)
	plt.legend(loc=0)
	plt.grid(True)
	"""
	plt.show()
	plt.clf()
	plt.close()
	
	return None

def read_data_from_trial_file(atten_value,trial_number):
   
   time = []
   a_uncorrected = []
   b_uncorrected = []
   c_uncorrected = []
   d_uncorrected = []
   e_uncorrected = []
   f_uncorrected = []
   a_corrected = []
   b_corrected = []
   c_corrected = []
   d_corrected = []
   e_corrected = []
   f_corrected = []
   
   if(atten_value==1000):
	   filename = "SNR 60_trial_&d.dat"%trial_number
   else:
	   filename = "Atten_%d_trial_%d.dat"%(atten_value,trial_number)
   # Read in data from file
   
   with open("/data/anita/hupe/Pulse_data/Atten_%1.2f/%s"%(atten_value,filename),'rb') as csvfile:
   #with open("testbench_analysis/data/sample_data.dat",'rb') as csvfile:
      reader = csv.reader(csvfile,delimiter=',', quotechar='|')
      row_num = 0
      for row in reader:
         if(row_num==0):
            row_num+=1
            continue
         time.append(row[0])
         a_uncorrected.append(float(row[1]))
         b_uncorrected.append(float(row[2]))
         c_uncorrected.append(float(row[3]))
         d_uncorrected.append(float(row[4]))
         e_uncorrected.append(float(row[5]))
         f_uncorrected.append(float(row[6]))
         
         a_corrected.append(float(row[7]))
         b_corrected.append(float(row[8]))
         c_corrected.append(float(row[9]))
         d_corrected.append(float(row[10]))
         e_corrected.append(float(row[11]))
         f_corrected.append(float(row[12]))
         
      return np.array(time),np.array(a_uncorrected),np.array(b_uncorrected),np.array(c_uncorrected),np.array(d_uncorrected),np.array(e_uncorrected),np.array(f_uncorrected),np.array(a_corrected),np.array(b_corrected),np.array(c_corrected),np.array(d_corrected),np.array(e_corrected),np.array(f_corrected)
     
def read_data_from_data_file(atten_value,num_samples_to_read,data_start):
   
	time = []
	a_uncorrected = []
	b_uncorrected = []
	c_uncorrected = []
	d_uncorrected = []
	e_uncorrected = []
	f_uncorrected = []
	a_corrected = []
	b_corrected = []
	c_corrected = []
	d_corrected = []
	e_corrected = []
	f_corrected = []
	# Read in data from file
   
   
	if(atten_value==100.0):
		filename = 'Atten_100.00_therm.dat'
	else:
		filename = 'Atten_%1.2f_imp.dat'%(atten_value)
   
	with open("/home/hupe.2/thesis_code/testbench_analysis/data/%s"%(filename),'rb') as csvfile:
		reader = csv.reader(csvfile,delimiter=',', quotechar='|')
		row_num = 0
		#print data_start*num_samples_to_read
		for row in reader:
			if(row_num<data_start*num_samples_to_read):
				row_num+=1
				continue
			elif(row_num>(num_samples_to_read+data_start*num_samples_to_read-1)):
				#print row_num
				break
			else:
				#print row_num
				time.append(float(row[0]))
				a_uncorrected.append(float(row[1]))
				b_uncorrected.append(float(row[2]))
				c_uncorrected.append(float(row[3]))
				d_uncorrected.append(float(row[4]))
				e_uncorrected.append(float(row[5]))
				f_uncorrected.append(float(row[6]))

				a_corrected.append(float(row[7]))
				b_corrected.append(float(row[8]))
				c_corrected.append(float(row[9]))
				d_corrected.append(float(row[10]))
				e_corrected.append(float(row[11]))
				f_corrected.append(float(row[12]))
				row_num+=1
         
	return np.array(time),np.array(a_uncorrected),np.array(b_uncorrected),np.array(c_uncorrected),np.array(d_uncorrected),np.array(e_uncorrected),np.array(f_uncorrected),np.array(a_corrected),np.array(b_corrected),np.array(c_corrected),np.array(d_corrected),np.array(e_corrected),np.array(f_corrected)
     	     
def read_impulsive_data_files(atten_value,data_start=0):
   
   time = []
   a_uncorrected = []
   b_uncorrected = []
   c_uncorrected = []
   d_uncorrected = []
   e_uncorrected = []
   f_uncorrected = []
   a_corrected = []
   b_corrected = []
   c_corrected = []
   d_corrected = []
   e_corrected = []
   f_corrected = []
   # Read in data from file
   with open("/home/hupe.2/thesis_code/testbench_analysis/data/Atten_%1.2f_imp.dat"%(atten_value),'rb') as csvfile:
      reader = csv.reader(csvfile,delimiter=',', quotechar='|')
      row_num = 0
      for row in reader:
         if(row_num==0):
            row_num+=1
            continue
         if(row_num>=data_start):
			 time.append(row[0])
			 a_uncorrected.append(float(row[1]))
			 b_uncorrected.append(float(row[2]))
			 c_uncorrected.append(float(row[3]))
			 d_uncorrected.append(float(row[4]))
			 e_uncorrected.append(float(row[5]))
			 f_uncorrected.append(float(row[6]))
			 
			 a_corrected.append(float(row[7]))
			 b_corrected.append(float(row[8]))
			 c_corrected.append(float(row[9]))
			 d_corrected.append(float(row[10]))
			 e_corrected.append(float(row[11]))
			 f_corrected.append(float(row[12]))
         
      return np.array(time),np.array(a_uncorrected),np.array(b_uncorrected),np.array(c_uncorrected),np.array(d_uncorrected),np.array(e_uncorrected),np.array(f_uncorrected),np.array(a_corrected),np.array(b_corrected),np.array(c_corrected),np.array(d_corrected),np.array(e_corrected),np.array(f_corrected)

def increase_impulse_frequency(a,b,c,d,e,f,first_pulse_position= 10,initial_freq=1000000.0,sample_frequency=2600000000.0,upscale_factor=81.25,debug=False):
	timestep = 1/sample_frequency

	new_freq = initial_freq*upscale_factor
	num_samp_before_peak = 10
	
	old_num_samples_between_pulses = sample_frequency/initial_freq
	
	new_num_samples_between_pulses = sample_frequency/new_freq
	if(debug==True):
		print old_num_samples_between_pulses
		print new_num_samples_between_pulses
		print len(a)
	
	# Use find impulse method to find each peak and cut it out
	first_peak_position = find_impulse_peak(a[:first_pulse_position*2],b[:first_pulse_position*2],c[:first_pulse_position*2],d[:first_pulse_position*2],e[:first_pulse_position*2],f[:first_pulse_position*2])
	first_peak_start = first_peak_position-num_samp_before_peak
	if(first_peak_start<0):
		first_peak_start=0
	first_peak_end = first_peak_start+new_num_samples_between_pulses
	if(first_peak_end>=len(a)-1):
		first_peak_end = len(a)-1
		first_peak_start = first_peak_end-new_num_samples_between_pulses
	new_a,new_b,new_c = a[first_peak_start:first_peak_end],b[first_peak_start:first_peak_end],c[first_peak_start:first_peak_end]
	new_d,new_e,new_f = d[first_peak_start:first_peak_end],e[first_peak_start:first_peak_end],f[first_peak_start:first_peak_end]
	
	if(debug==True):
		print "First Peak Window Start %d (%d ns)"%(0,0)
		print "First Peak Window End %d (%d ns)"%((first_pulse_position+100),((first_pulse_position+100)*timestep*10**9))
		print "First Peak Start %d (%d ns)"%(first_peak_start,first_peak_start*timestep*10**9)
		print "First Peak End %d (%d ns)"%(first_peak_end,first_peak_end*timestep*10**9)
		print len(new_a)
	
	second_peak_position = find_impulse_peak(a[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],b[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],c[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],d[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],e[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],f[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5])
	second_peak_start = first_peak_end+second_peak_position-num_samp_before_peak
	if(second_peak_start<0):
		second_peak_start=0
	second_peak_end = second_peak_start+new_num_samples_between_pulses
	if(second_peak_end>=len(a)-1):
		second_peak_end = len(a)-1
		second_peak_start = second_peak_end-new_num_samples_between_pulses
	#new_a,new_b,new_c = a[second_peak_start:second_peak_end],b[second_peak_start:second_peak_end],c[second_peak_start:second_peak_end]
	new_a,new_b,new_c = np.concatenate((new_a,a[second_peak_start:second_peak_end])),np.concatenate((new_b,b[second_peak_start:second_peak_end])),np.concatenate((new_c,c[second_peak_start:second_peak_end]))
	new_d,new_e,new_f = np.concatenate((new_d,d[second_peak_start:second_peak_end])),np.concatenate((new_e,e[second_peak_start:second_peak_end])),np.concatenate((new_f,f[second_peak_start:second_peak_end]))
	
	if(debug==True):
		print "Second Peak Window Start %d (%d ns)"%(first_peak_end,first_peak_end*timestep*10**9)
		print "Second Peak Window End %d (%d ns)"%((first_peak_end+old_num_samples_between_pulses*1.5),(first_peak_end+old_num_samples_between_pulses*1.5)*timestep*10**9)
		print "Second Peak Start %d (%d ns)"%(second_peak_start,second_peak_start*timestep*10**9)
		print "Second Peak End %d (%d ns)"%(second_peak_end,second_peak_end*timestep*10**9)
		print len(new_a)
	
	
	third_peak_position = find_impulse_peak(a[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],b[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],c[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],d[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],e[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],f[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5])
	third_peak_start = second_peak_end+third_peak_position-num_samp_before_peak
	if(third_peak_start<0):
		third_peak_start=0
	third_peak_end = third_peak_start+new_num_samples_between_pulses
	if(third_peak_end>=len(a)-1):
		third_peak_end = len(a)-1
		third_peak_start = third_peak_end-new_num_samples_between_pulses
	#new_a,new_b,new_c = a[third_peak_start:third_peak_end],b[third_peak_start:third_peak_end],c[third_peak_start:third_peak_end]
	new_a,new_b,new_c = np.concatenate((new_a,a[third_peak_start:third_peak_end])),np.concatenate((new_b,b[third_peak_start:third_peak_end])),np.concatenate((new_c,c[third_peak_start:third_peak_end]))
	new_d,new_e,new_f = np.concatenate((new_d,d[third_peak_start:third_peak_end])),np.concatenate((new_e,e[third_peak_start:third_peak_end])),np.concatenate((new_f,f[third_peak_start:third_peak_end]))
	if(debug==True):
		print "Third Peak Window Start %d (%d ns)"%(second_peak_end,second_peak_end*timestep*10**9)
		print "Third Peak Window End %d (%d ns)"%((second_peak_end+old_num_samples_between_pulses*1.5),(second_peak_end+old_num_samples_between_pulses*1.5)*timestep*10**9)
		print "Third Peak Start %d (%d ns)"%(third_peak_start,third_peak_start*timestep*10**9)
		print "Third Peak End %d (%d ns)"%(third_peak_end,third_peak_end*timestep*10**9)
		print len(new_a)
	
	
	new_t = np.linspace(0,len(new_a)*timestep,len(new_a))
	
	return new_a, new_b,new_c, new_t
	
	
def form_pure_impulse_file(a,b,c,d,e,f,sample_window=80,first_pulse_position=1000,initial_freq=1000000.0,sample_frequency=2600000000.0,debug=False):
	timestep = 1/sample_frequency

	
	num_samp_before_peak = 40
	
	old_num_samples_between_pulses = sample_frequency/initial_freq
	
	new_num_samples_between_pulses = 80
	"""
	if(debug==True):
		print old_num_samples_between_pulses
		print new_num_samples_between_pulses
		print len(a)
	"""
	
	low_bin_factor = 0.9
	high_bin_factor = 1.1
	
	# Use find impulse method to find each peak and cut it out
	first_peak_position = find_impulse_peak(a[:first_pulse_position*2],b[:first_pulse_position*2],c[:first_pulse_position*2],d[:first_pulse_position*2],e[:first_pulse_position*2],f[:first_pulse_position*2],debug=debug)
	first_peak_start = first_peak_position-num_samp_before_peak
	if(first_peak_start<0):
		first_peak_start=0
	first_peak_end = first_peak_start+new_num_samples_between_pulses
	if(first_peak_end>=len(a)-1):
		first_peak_end = len(a)-1
		first_peak_start = first_peak_end-new_num_samples_between_pulses
	new_a,new_b,new_c =a[first_peak_start:first_peak_end],b[first_peak_start:first_peak_end],c[first_peak_start:first_peak_end]
	new_d,new_e,new_f = d[first_peak_start:first_peak_end],e[first_peak_start:first_peak_end],f[first_peak_start:first_peak_end]
	"""
	if(debug==True):
		print "First Peak Window Start %d (%d ns)"%(0,0)
		print "First Peak Window End %d (%d ns)"%((first_pulse_position+100),((first_pulse_position+100)*timestep*10**9))
		print "First Peak Start %d (%d ns)"%(first_peak_start,first_peak_start*timestep*10**9)
		print "First Peak End %d (%d ns)"%(first_peak_end,first_peak_end*timestep*10**9)
		#print len(new_a)
	"""
	#second_peak_position = find_impulse_peak(a[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],b[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],c[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],d[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],e[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],f[first_peak_end:first_peak_end+old_num_samples_between_pulses*1.5],debug=debug)
	#second_peak_position = find_impulse_peak(a[int((first_peak_position+old_num_samples_between_pulses)*low_bin_factor):int((first_peak_position+old_num_samples_between_pulses)*high_bin_factor)],b[int((first_peak_position+old_num_samples_between_pulses)*low_bin_factor):int((first_peak_position+old_num_samples_between_pulses)*high_bin_factor)],c[int((first_peak_position+old_num_samples_between_pulses)*low_bin_factor):int((first_peak_position+old_num_samples_between_pulses)*high_bin_factor)],d[int((first_peak_position+old_num_samples_between_pulses)*low_bin_factor):int((first_peak_position+old_num_samples_between_pulses)*high_bin_factor)],e[int((first_peak_position+old_num_samples_between_pulses)*low_bin_factor):int((first_peak_position+old_num_samples_between_pulses)*high_bin_factor)],f[int((first_peak_position+old_num_samples_between_pulses)*low_bin_factor):int((first_peak_position+old_num_samples_between_pulses)*high_bin_factor)],debug=debug)
	#second_peak_start = first_peak_end+second_peak_position-num_samp_before_peak
	second_peak_start = first_peak_start+old_num_samples_between_pulses
	if(second_peak_start<0):
		second_peak_start=0
	second_peak_end = second_peak_start+new_num_samples_between_pulses
	if(second_peak_end>=len(a)-1):
		second_peak_end = len(a)-1
		second_peak_start = second_peak_end-new_num_samples_between_pulses
	#new_a,new_b,new_c = a[second_peak_start:second_peak_end],b[second_peak_start:second_peak_end],c[second_peak_start:second_peak_end]
	new_a,new_b,new_c = np.concatenate((new_a,a[second_peak_start:second_peak_end])),np.concatenate((new_b,b[second_peak_start:second_peak_end])),np.concatenate((new_c,c[second_peak_start:second_peak_end]))
	new_d,new_e,new_f = np.concatenate((new_d,d[second_peak_start:second_peak_end])),np.concatenate((new_e,e[second_peak_start:second_peak_end])),np.concatenate((new_f,f[second_peak_start:second_peak_end]))
	
	if(debug==True):
		#print "Second Peak Window Start %d (%d ns)"%(first_peak_end,first_peak_end*timestep*10**9)
		#print "Second Peak Window End %d (%d ns)"%((first_peak_end+old_num_samples_between_pulses*1.5),(first_peak_end+old_num_samples_between_pulses*1.5)*timestep*10**9)
		#print "Second Peak Start %d (%d ns)"%(second_peak_start,second_peak_start*timestep*10**9)
		#print "Second Peak End %d (%d ns)"%(second_peak_end,second_peak_end*timestep*10**9)
		print "AB Period: %d (%d ns)"%(second_peak_start-first_peak_start,(second_peak_start-first_peak_start)*timestep*10**9)
		#print len(new_a)
	
	#third_peak_position = find_impulse_peak(a[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],b[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],c[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],d[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],e[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],f[second_peak_end:second_peak_end+old_num_samples_between_pulses*1.5],debug=debug)
	#third_peak_position = find_impulse_peak(a[int((2*old_num_samples_between_pulses+first_peak_position)*low_bin_factor):int((2*old_num_samples_between_pulses+first_peak_position)*high_bin_factor)],b[int((2*old_num_samples_between_pulses+first_peak_position)*low_bin_factor):int((2*old_num_samples_between_pulses+first_peak_position)*high_bin_factor)],c[int((2*old_num_samples_between_pulses+first_peak_position)*low_bin_factor):int((2*old_num_samples_between_pulses+first_peak_position)*high_bin_factor)],d[int((2*old_num_samples_between_pulses+first_peak_position)*low_bin_factor):int((2*old_num_samples_between_pulses+first_peak_position)*high_bin_factor)],e[int((2*old_num_samples_between_pulses+first_peak_position)*low_bin_factor):int((2*old_num_samples_between_pulses+first_peak_position)*high_bin_factor)],f[int((2*old_num_samples_between_pulses+first_peak_position)*low_bin_factor):int((2*old_num_samples_between_pulses+first_peak_position)*high_bin_factor)],debug=debug)
	#third_peak_start = second_peak_end+third_peak_position-num_samp_before_peak
	third_peak_start = second_peak_start+old_num_samples_between_pulses
	if(third_peak_start<0):
		third_peak_start=0
	third_peak_end = third_peak_start+new_num_samples_between_pulses
	if(third_peak_end>=len(a)-1):
		third_peak_end = len(a)-1
		third_peak_start = third_peak_end-new_num_samples_between_pulses
	#new_a,new_b,new_c = a[third_peak_start:third_peak_end],b[third_peak_start:third_peak_end],c[third_peak_start:third_peak_end]
	new_a,new_b,new_c = np.concatenate((new_a,a[third_peak_start:third_peak_end])),np.concatenate((new_b,b[third_peak_start:third_peak_end])),np.concatenate((new_c,c[third_peak_start:third_peak_end]))
	new_d,new_e,new_f = np.concatenate((new_d,d[third_peak_start:third_peak_end])),np.concatenate((new_e,e[third_peak_start:third_peak_end])),np.concatenate((new_f,f[third_peak_start:third_peak_end]))
	
	if(debug==True):
		#print "Third Peak Window Start %d (%d ns)"%(second_peak_end,second_peak_end*timestep*10**9)
		#print "Third Peak Window End %d (%d ns)"%((second_peak_end+old_num_samples_between_pulses*1.5),(second_peak_end+old_num_samples_between_pulses*1.5)*timestep*10**9)
		#print "Third Peak Start %d (%d ns)"%(third_peak_start,third_peak_start*timestep*10**9)
		#print "Third Peak End %d (%d ns)"%(third_peak_end,third_peak_end*timestep*10**9)
		print "BC Period: %d (%d ns)"%(third_peak_start-second_peak_start,(third_peak_start-second_peak_start)*timestep*10**9)
		#print len(new_a)
	
	new_t = np.linspace(0,len(new_a)*timestep,len(new_a))
	
	return new_a, new_b,new_c,new_d,new_e,new_f, new_t
	
def find_impulse_peak(a,b,c,d,e,f,debug=False):
   abc_signal = np.add(a,np.add(b,c))
   def_signal = np.add(d,np.add(e,f))
   abcdef_signal = np.add(abc_signal,def_signal)
   #abs_signal = np.square(signal)
   peak_element = np.argmin(abcdef_signal)
   if(debug):
	   #print "A peak: %.2f"%a[peak_element]
	   #print "B peak: %.2f"%b[peak_element]
	   #print "C peak: %.2f"%c[peak_element]
	   #print "D peak: %.2f"%d[peak_element]
	   #print "E peak: %.2f"%e[peak_element]
	   #print "F peak: %.2f"%f[peak_element]
	   print "Sum peak: %.2f"%abcdef_signal[peak_element]
  
   
   return peak_element
     
def align_phi_sector(a,b,c,d,e,f,impulse_position = 1000):
   
   a_peak_element = find_impulse_peak(a,b,c,d,e,f)
   
   num_samples_to_shift = impulse_position - a_peak_element
   #print "Num to shift: %d"%num_samples_to_shift
   a = np.roll(a,num_samples_to_shift)
   b = np.roll(b,num_samples_to_shift)
   c = np.roll(c,num_samples_to_shift)
   d = np.roll(d,num_samples_to_shift)
   e = np.roll(e,num_samples_to_shift)
   f = np.roll(f,num_samples_to_shift)
   
   
   return a,b,c,d,e,f

def write_to_file(file_loc,filename,array):
		
	with open((str(file_loc)+'/'+str(filename)),'ab') as f:
		writer = csv.writer(f)
		for i in range(len(array)):
			writer.writerow(array[i])	
	
		
	return None
	
	

 
def build_data_file(input_file_atten,num_files,impulse_file=True):

	print "Reading INL corrections"
	#Do INL correction on uncorrected data
	a_offsets = get_individual_sample_offsets(0,0)
	b_offsets = get_individual_sample_offsets(0,1)
	c_offsets = get_individual_sample_offsets(0,2)
	d_offsets = get_individual_sample_offsets(0,4)
	e_offsets = get_individual_sample_offsets(0,5)
	f_offsets = get_individual_sample_offsets(0,6)
			
	timestep = 1.0/2600000000.0
	print "Starting attenuation: %1.2f"%input_file_atten
	for i in range(num_files):
		if(i % 100 == 0):
			print "Reading trial #%d"%i
		t,au,bu,cu,du,eu,fu, a,b,c,d,e,f = read_data_from_trial_file(input_file_atten,i)
		if(impulse_file==True):
			if(i % 100 == 0):
				print "Building Impulse File"
			#print "Correcting INL"
			#Do INL correction (but not glitch correction
			a = np.array(correct_individual_samples(au+3.5,a_offsets))-3.5
			b = np.array(correct_individual_samples(bu+3.5,b_offsets))-3.5
			c = np.array(correct_individual_samples(cu+3.5,c_offsets))-3.5
			d = np.array(correct_individual_samples(du+3.5,d_offsets))-3.5
			e = np.array(correct_individual_samples(eu+3.5,e_offsets))-3.5
			f = np.array(correct_individual_samples(fu+3.5,f_offsets))-3.5
			
			#print "Aligning phi sectors"
			# Align each phi sector
			au,bu,cu,du,eu,fu = align_phi_sector(au,bu,cu,du,eu,fu)
			a,b,c,d,e,f = align_phi_sector(a,b,c,d,e,f)
			#print len(au)
			#print len(du)
		
			#if(impulse_file==True):
			# Artificially Increase the sample frequency
			#au,bu,cu,t = increase_impulse_frequency(au,bu,cu,upscale_factor=new_frequency)
			#du,eu,fu,t = increase_impulse_frequency(du,eu,fu,upscale_factor=new_frequency)
			#a,b,c,t = increase_impulse_frequency(a,b,c,upscale_factor=new_frequency)
			#d,e,f,t = increase_impulse_frequency(d,e,f,upscale_factor=new_frequency)
			
			#print "Finding impulses"
			au,bu,cu,du,eu,fu,t = form_pure_impulse_file(au,bu,cu,du,eu,fu)
			a,b,c,d,e,f,t = form_pure_impulse_file(a,b,c,d,e,f)
			#print len(au)
			#print len(du)
			
		
		num_samples = len(a)
		time = np.linspace(i*num_samples*timestep,(i+1)*num_samples*timestep,num_samples)
		
		
		# Convert to one matrix
		write_matrix = np.zeros((13,num_samples))
		#write_matrix = [0]*13
		write_matrix[0] = time
		write_matrix[1] = au
		write_matrix[2] = bu
		write_matrix[3] = cu
		write_matrix[4] = du
		write_matrix[5] = eu
		write_matrix[6] = fu
		
		write_matrix[7] = a
		write_matrix[8] = b
		write_matrix[9] = c
		write_matrix[10] = d
		write_matrix[11] = e
		write_matrix[12] = f
		write_matrix = np.transpose(write_matrix)

		# Write to one large file
		#write_to_file('testbench_analysis/data','test_concat_data.dat',write_matrix)
		if(impulse_file==True):
			write_to_file('testbench_analysis/data','Atten_%1.2f_imp_2.dat'%(input_file_atten),write_matrix)
		else:
			write_to_file('testbench_analysis/data','Atten_%1.2f_therm.dat'%(input_file_atten),write_matrix)
		
	return None
	
   
   
   
def delay_antennas(b,c,b_delay,c_delay):
   
   if(b_delay>=1 or c_delay<=-1): 
	   b=np.roll(b,b_delay)
   
   if(c_delay>=1 or c_delay<=-1): 
	   c=np.roll(c,c_delay)
   
   return b,c
   
   
def find_nearest(array,value):
   idx = (np.abs(array-value)).argmin()
   return idx,array[idx]

def fit_exponential_tail(x,y,debug=False):
   
   max_value = np.amax(y)
   
   end = len(y)
   #Find 1/100 max point
   for i in range(0,len(y)):
      if(y[i]<0.001*max_value):
         fit_start = i
         break
   for j in range(fit_start,len(y)):
      if(y[j]==0.0):
         fit_end =j
         break
   if(debug==True):
	   print max_value
	   print fit_start
	   print fit_end
	   #plt.plot(x,y)
	   print y[fit_start:fit_end]
	   print x[fit_start:fit_end]
   lin_y = np.log(y[fit_start:fit_end])
   #plt.plot(x[fit_start:fit_end],lin_y,color="r")
   #plt.show()
   #print lin_y
   #a,b = np.polyfit(x[fit_start:fit_end],lin_y,1)
   #fit = a*np.array(x)+b
   a,b.c = np.polyfit(x[fit_start:fit_end],lin_y,2)
   fit = a*np.square(x)+b*np.array(x)+c
   exp_fit = np.exp(fit)
   #print exp_fit[fit_end:]
   #plt.semilogy(x[fit_end:],exp_fit[fit_end:],color='b',linestyle="--")
   #plt.show()
   return fit_end,a,b


def sigmoid(p,x):
	x0,y0,c,k=p
	y = c / (1+np.exp(-k*(x-x0))) +y0
	return y
	
def residuals(p,x,y):
	return y - sigmoid(p,x)
	
def resize(arr,lower=0.0,upper=1.0):
	arr=arr.copy()
	if (lower>upper): 
		lower,upper=upper, lower
	arr -= arr.min()
	arr *= (upper-lower)/arr.max()
	arr += lower
	return arr

def make_thermal_power_data_file():
	import matplotlib.pyplot as plt
	import csv
	
	t,au,bu,cu,du,eu,fu,a,b,c,d,e,f = read_data_from_data_file(100.0,320000,0)
	
	num_blocks = len(t)/32
	power = np.zeros(num_blocks)
	filename = "thermal_power.dat"
	file_loc = "/home/hupe.2/thesis_code/testbench_analysis/output/"
	f=open(file_loc+filename,'w')
	
	for i in range(num_blocks):
	
		start = i*32
		end = start+32
	
		power[i] = np.sum(np.square(np.add(a[start:end],np.add(b[start:end],c[start:end]))))
		f.write(str(power[i])+'\n')
	plt.hist(power,bins=num_blocks*0.002)
	plt.show()
	
	
	
	f.close()
	
		
	
	return power
	
	
def norm(x, mean, sd):
	norm = []
	for i in range(x.size):
		norm += [1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x[i] - mean)**2/(2*sd**2))]
	return np.array(norm)
	
def res(p, y, x):
	m, dm, sd1, sd2 = p
	m1 = m
	m2 = m1 + dm
	y_fit = norm(x, m1, sd1) + norm(x, m2, sd2)
	err = y - y_fit
	return err
	
	
