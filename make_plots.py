#!/usr/bin/env python

if __name__ == '__main__':
	import numpy as np
	import matplotlib.pyplot as plt
	import random
	from noise import generate_noise
	from digitizer import digitize
	from cw import generate_cw
	from sum_correlator import sum_correlate
	from anita_filter import butter_bandpass, butter_bandpass_filter
	from impulse import impulse_gen
	from scipy.stats import norm

	sample_frequency = 2600000000.0
	sample_length = 80
	impulse_position = 5
	upsample = 10
	draw_flag = 1
	output_dir = 'output/'
	SNR = 6
	noise_sigma = 32.0
	save_plot_flag = True
	cw_flag = False
	carrier_frequency=260000000.0    # Hz
	modulation_frequency= 1.0  # Hz
	cw_rms = 1.0*noise_sigma #Peak amplitude in mV
	digitization_factor = 32.0

	upsample_length = upsample*sample_length

	impulse_sample = np.zeros(sample_length)
	impulse_upsample = np.zeros(upsample_length)
	impulse_downsample = np.zeros(sample_length)
	#impulse_upsample_fit_poly = np.zeros(poly_index)
	impulse_upsample_fit = np.zeros(upsample_length)
	impulse_noise_sample = np.zeros(sample_length)
	digitized_sample = np.zeros(sample_length)
	noise_sample = np.zeros(sample_length)
	cw_sample = np.zeros(sample_length)
	#noise_cw_sample = np.zeros(sample_length)

	time = np.linspace(0.0,((sample_length*(10**9))/sample_frequency), sample_length)
	impulse_sample = impulse_gen(sample_length,impulse_position,upsample,sample_frequency,draw_flag=draw_flag,output_dir=output_dir)
	
	
	"""
	upsample_time = np.linspace(0.0,((upsample_length*(10**9))/(sample_frequency*upsample)), upsample_length)
	plt.figure(2,figsize=(16,8))
	plt.clf()
	plt.plot(upsample_time,impulse_upsample[0:upsample_length])
	plt.xlabel("Time [ns]")
	plt.ylabel("Amplitude [unitless]")
	plt.title("Upsampled Impulse Simulation")
	if (save_plot_flag):
		plt.savefig("plots/upsample_impulse_sample.png")
	"""
	signal_amp = (2*SNR*noise_sigma)
	difference=np.amax(impulse_sample)-np.amin(impulse_sample) # Get peak to peak voltage
	impulse_sample *= (1/difference) # Normalize input
	impulse_sample *= signal_amp # Amplify
	plt.figure(1,figsize=(16,8))
	plt.clf()
	plt.plot(time,impulse_sample[0:sample_length])
	plt.xlabel("Time [ns]")
	plt.ylabel("Amplitude [mV]")
	plt.title("Impulse Simulation")
	if (save_plot_flag):
		plt.savefig("plots/impulse_sample.png")
	b_impulse_sample = np.roll(impulse_sample,-15)
	c_impulse_sample = np.roll(impulse_sample,-17)
	#print b_impulse_sample
	#print c_impulse_sample

	noise_sample = generate_noise(sample_length=sample_length,noise_sigma=noise_sigma,filter_flag=0)
	b_noise_sample = generate_noise(sample_length=sample_length,noise_sigma=noise_sigma,filter_flag=0)
	c_noise_sample = generate_noise(sample_length=sample_length,noise_sigma=noise_sigma,filter_flag=0)
	plt.figure(3,figsize=(16,8))
	plt.clf()
	plt.plot(time,noise_sample)
	plt.xlabel("Time [ns]")
	plt.ylabel("Voltage [mV]")
	plt.title("Band Limited Noise Sample")
	if (save_plot_flag):
		plt.savefig("plots/noise_sample.png")
		
	plt.figure(4,figsize=(16,8))
	plt.clf()
	test_noise = generate_noise(sample_length=sample_length*1000,noise_sigma=noise_sigma,filter_flag=False)
	plt.hist(test_noise,bins=100,normed=True,label="N = 80,000")
	mu,std = np.mean(test_noise),np.sqrt(np.var(test_noise))
	xmin,xmax = plt.xlim()
	#print mu,std
	x = np.linspace(xmin,xmax,100)
	p = norm.pdf(x,mu,std)
	#print p
	plt.plot(x,p,'k',label = "Fit results: mu = %.2f, std = %.2f" % (mu,std))
	plt.xlabel("Voltage [mV}")
	plt.ylabel("Counts")
	plt.title("Simulated Thermal Noise Distribution")
	plt.legend()
	if (save_plot_flag):
		plt.savefig("plots/thermal_noise_distribution.png")
	
	
	if (cw_flag):
		cw_sample = generate_cw(sample_length=sample_length,upsample=upsample,sample_frequency=sample_frequency,carrier_frequency=carrier_frequency,modulation_frequency=modulation_frequency,cw_rms=cw_rms,filter_flag=1)
		b_cw_sample = cw_sample#generate_cw(sample_length=sample_length,sample_frequency=sample_frequency,carrier_frequency=carrier_frequency,modulation_frequency=modulation_frequency,peak_amplitude=peak_amplitude,filter_flag=1)
		c_cw_sample = cw_sample#generate_cw(sample_length=sample_length,sample_frequency=sample_frequency,carrier_frequency=carrier_frequency,modulation_frequency=modulation_frequency,peak_amplitude=peak_amplitude,filter_flag=1)
		plt.figure(4,figsize=(16,8))
		plt.clf()
		plt.plot(time,cw_sample)
		plt.xlabel("Time [ns]")
		plt.ylabel("Voltage [mV]")
		plt.title("Carrier Wave Sample (Peak Amp: "+str(peak_amplitude/noise_sigma)+" x Noise RMS)")
		
		if (save_plot_flag):
			plt.savefig("plots/cw_sample.png")
		 
		plt.figure(5,figsize=(16,8))
		plt.clf()
		plt.plot(time,noise_sample)
		plt.xlabel("Time [ns]")
		plt.ylabel("Voltage [mV]")
		plt.title("Noise+CW Sample")
		if (save_plot_flag):
			plt.savefig("plots/noise_cw_sample.png")
		
	impulse_noise_sample = np.add(impulse_sample,noise_sample)
	b_impulse_noise_sample = np.add(b_impulse_sample,b_noise_sample)
	c_impulse_noise_sample = np.add(c_impulse_sample,c_noise_sample)
	plt.figure(6,figsize=(16,8))
	plt.clf()
	plt.plot(time,impulse_noise_sample)
	plt.xlabel("Time [ns]")
	plt.ylabel("Voltage [mV]")
	plt.title("Impulse(SNR "+str(SNR)+") + Noise Sample(RMS "+str(noise_sigma)+"mV)")
	if (save_plot_flag):
		plt.savefig("plots/impulse_noise_sample.png")
		
	plt.figure(7,figsize=(16,8))
	plt.clf()
	plt.plot(time,impulse_noise_sample,time,b_impulse_noise_sample,time,c_impulse_noise_sample)
	plt.xlabel("Time [ns]")
	plt.ylabel("Voltage [mV]")
	plt.title("Impulse(SNR "+str(SNR)+") + Noise Sample(RMS "+str(noise_sigma)+"mV)")
	if (save_plot_flag):
		plt.savefig("plots/3ch_impulse_noise_sample.png")


	digitized_sample = digitize(impulse_noise_sample,sample_length,num_bits=3,digitization_factor=digitization_factor)
	b_digitized_sample = digitize(b_impulse_noise_sample,sample_length,num_bits=3,digitization_factor=digitization_factor)
	c_digitized_sample = digitize(c_impulse_noise_sample,sample_length,num_bits=3,digitization_factor=digitization_factor)
	
	plt.figure(9,figsize=(16,8))
	plt.clf()
	plt.plot(time,digitized_sample)
	plt.xlabel("Time [ns]")
	plt.ylabel("Digital Levels [DAC Units]")
	plt.title("Digitized Impulse + Noise Sample")
	plt.ylim(-3.5,3.5)
	if (save_plot_flag):
		plt.savefig("plots/digitized_impulse_noise_sample.png")
		
	plt.figure(10,figsize=(16,8))
	plt.clf()
	plt.plot(time,digitized_sample,time,b_digitized_sample,time,c_digitized_sample)
	plt.xlabel("Time [ns]")
	plt.ylabel("Digital Levels [DAC Units]")
	plt.title("Digitized Impulse + Noise Sample")
	plt.ylim(-3.5,3.5)
	if (save_plot_flag):
		plt.savefig("plots/3_ch_digitized_impulse_noise_sample.png")
		
		
	plt.figure(11,figsize=(16,8))
	plt.clf()
	plt.plot(time,digitized_sample,time,np.roll(b_digitized_sample,15),time,np.roll(c_digitized_sample,17))
	plt.xlabel("Time [ns]")
	plt.ylabel("Digital Levels [DAC Units]")
	plt.title("Digitized Impulse + Noise Sample Aligned")
	plt.ylim(-3.5,3.5)
	if (save_plot_flag):
		plt.savefig("plots/aligned_3_ch_digitized_impulse_noise_sample.png")
		
		
	
	##############################
	#print best_delays
	#time = np.linspace(0.0,((num_samples*(10**9))/2600000000.0), TISC_sample_length)
	#print len(time)
	a_start_pos = 37
	b_start_pos = 37-15
	c_start_pos = 37-17
	sum_length=32
	add_AB = np.add(digitized_sample[a_start_pos:a_start_pos+sum_length],b_digitized_sample[b_start_pos:b_start_pos+sum_length])
	bad_add_AB = np.add(digitized_sample[a_start_pos:a_start_pos+sum_length],b_digitized_sample[a_start_pos:a_start_pos+sum_length])
	#print len(add_AB)
	#print add_AB
	add_ABC = np.add(add_AB,c_digitized_sample[c_start_pos:c_start_pos+sum_length])
	bad_add_ABC = np.add(bad_add_AB,c_digitized_sample[a_start_pos:a_start_pos+sum_length])
	#print len(add_ABC)
	#print len(add_ABC)
	#print add_ABC
	square_ABC = add_ABC**2
	bad_square_ABC = bad_add_ABC**2
	
	sum_corr = np.sum(square_ABC)
	bad_sum_corr = np.sum(bad_square_ABC)
	print sum_corr
	print bad_sum_corr

	plt.figure(12,figsize=(16,8))
	plt.clf()
	plt.plot(time[a_start_pos:a_start_pos+sum_length],add_ABC)
	plt.xlabel("Time [ns]")
	plt.ylabel("Signal [DAC Units]")
	plt.title("ABC Added Signal")
	if (1):
	  plt.savefig("plots/abc_add.png")
	  
	plt.figure(13,figsize=(16,8))
	plt.clf()
	plt.plot(time[a_start_pos:a_start_pos+sum_length],square_ABC)
	plt.xlabel("Time [ns]")
	plt.ylabel("Signal [DAC Units]")
	plt.title("ABC^2 Added Signal")
	if (1):
	  plt.savefig("plots/abc_add_square.png")
	plt.show()
	############################


	#plt.show()
