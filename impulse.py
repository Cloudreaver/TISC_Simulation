#!/usr/bin/env python

def impulse_gen(sim_param):
    # Input defaults are set for TISC impulse response

    import numpy as np
    if(sim_param.draw_flag):
        import matplotlib.pyplot as plt
    from scipy.signal import freqz,resample
    from scipy import ifft
    from anita_filter import butter_bandpass
    import csv
    
    time = []
    sample_signal_temp = []
    low_corner_freq =   180000000.0
    high_corner_freq = 1200000000.0
    
    with open("/home/hupe.2/thesis_code/anita_II_impulse_response.dat",'rb') as csvfile:
		reader = csv.reader(csvfile,delimiter=' ', quotechar='|')
		for row in reader:
			time.append(float(row[0])*10**9)
			sample_signal_temp.append(float(row[1]))

    #print np.argmax(np.abs(sample_signal_temp))
    sample_signal_temp = np.roll(sample_signal_temp,-np.argmax(np.abs(sample_signal_temp)))
    #delta_t = t[1]-t[0]
    #upsample = delta_t*freq_response
    #print upsample
    # Get number of samples in upscaled signal
    num_upsamples = int(sim_param.num_samples*sim_param.upsample)

    signal = np.zeros(sim_param.num_samples)
    signal_temp = np.zeros(sim_param.num_samples)
    old_upsample_signal_temp = np.zeros(num_upsamples)
    upsample_signal = np.zeros(num_upsamples)
    old_upsample_signal = np.zeros(num_upsamples)
    
    # Find Nyquist Frequency
    #nyquist_freq = freq_response/2.0
    
    # Get time axis
    #time = np.linspace(0.0,(num_samples/freq_response)*(10**9), num_samples)
    upsample_time = np.linspace(0.0,(sim_param.num_samples/sim_param.sample_frequency)*(10**9), num_upsamples)

    b, a = butter_bandpass(low_corner_freq, high_corner_freq, order=8,fs=2600000000.0)
    w, h = freqz(b, a, worN=sim_param.num_samples)
    
    # Transform bandpass filter into time domain to get impulse signal
    #old_upsample_signal_temp = ifft(h,num_upsamples).real
        
    upsample_signal_temp = resample(sample_signal_temp,num_upsamples)    
    
    # Delay impulse (But keep start after 37 to allow all delays to run)
    random_delay = int(np.random.uniform()*16)+37

    upsample_signal = np.roll(upsample_signal_temp,random_delay*sim_param.upsample)
    #old_upsample_signal = np.roll(old_upsample_signal_temp,random_delay*upsample)
    #upsample_signal[random_delay*upsample:num_upsamples] = upsample_signal_temp[0:num_upsamples-random_delay*upsample]
    
    sample_period = upsample_time[1]-upsample_time[0]
    phase = int(np.random.uniform()*sim_param.upsample)
    
    for t in range(0,sim_param.num_samples):
        signal[t] = upsample_signal[t*sim_param.upsample+phase]

    # Plot the frequency response of the antenna
    
    if(sim_param.draw_flag):
        plt.figure(1,figsize=(8,4))
        plt.clf()
        plt.semilogy(((sim_param.sample_frequency * 0.5 / np.pi) * w)/(10**9), abs(h), label="order = %d" % order)
        plt.axis([0,1.4,0.01,10])
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Gain')
        plt.title('Simulated Bandpass Filter')
        plt.grid(True)
    
    if (sim_param.draw_flag==1):
        plt.figure(2,figsize=(8,4))
        plt.plot(upsample_time,1000.0*upsample_signal[0:num_upsamples])
        plt.axis([0,35,-1500,1500])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("26 GSa/sec Impulse Simulation")
        
    #if (draw_flag==1):
        #plt.figure(3,figsize=(8,4))
        #plt.plot(upsample_time,1000.0*old_upsample_signal_temp[0:num_upsamples])
        #plt.xlabel("Time [ns]")
        #plt.ylabel("Voltage [mV]")
        #plt.title("Upsampled Impulse Simulation")
    
    if (sim_param.draw_flag==1):
        plt.figure(4,figsize=(8,4))
        plt.plot(time[0:sim_param.num_samples],1000.0*signal[0:sim_param.num_samples])
        plt.axis([0,35,-1500,1500])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("2.6 GSa/sec Impulse Simulation")
        plt.show()

    return signal


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from simulation_parameters import sim_param
    
    sample_frequency = 2600000000.0
    sample_length = 80
    impulse_position = 0
    upsample = 10
    draw_flag = 0
    
    upsample_length = sim_param.upsample*sim_param.num_samples
    
    impulse_sample = np.zeros(sim_param.num_samples)
    impulse_upsample_fit = np.zeros(upsample_length)
    
    time = np.linspace(0.0,((sim_param.num_samples*(10**9))/sim_param.sample_frequency), sim_param.num_samples)
    impulse_sample = impulse_gen(sim_param)
    #print impulse_sample
    print np.argmax(impulse_sample)
    plt.figure(1)
    plt.clf()
    plt.plot(time,impulse_sample[0:sim_param.num_samples])
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title("Impulse Simulation")
    #plt.show()
