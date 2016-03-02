#!/usr/bin/env python

def impulse_gen(num_samples,delay,upsample=10,freq_response=2600000000.0,low_corner_freq=180000000.0,high_corner_freq=1200000000.0,order=8,draw_flag=0,output_dir='output/'):
    # Input defaults are set for TISC impulse response

    import numpy as np
    if(draw_flag):
        import matplotlib.pyplot as plt
    from scipy.signal import freqz,resample
    from scipy import ifft
    from anita_filter import butter_bandpass
    import csv
    
    time = []
    sample_signal_temp = []
    
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
    num_upsamples = int(num_samples*upsample)

    signal = np.zeros(num_samples)
    signal_temp = np.zeros(num_samples)
    old_upsample_signal_temp = np.zeros(num_upsamples)
    upsample_signal = np.zeros(num_upsamples)
    old_upsample_signal = np.zeros(num_upsamples)
    
    # Find Nyquist Frequency
    #nyquist_freq = freq_response/2.0
    
    # Get time axis
    #time = np.linspace(0.0,(num_samples/freq_response)*(10**9), num_samples)
    upsample_time = np.linspace(0.0,(num_samples/freq_response)*(10**9), num_upsamples)

    b, a = butter_bandpass(low_corner_freq, high_corner_freq, order=8,fs=2600000000.0)
    w, h = freqz(b, a, worN=num_samples)
    
    # Transform bandpass filter into time domain to get impulse signal
    #old_upsample_signal_temp = ifft(h,num_upsamples).real
        
    upsample_signal_temp = resample(sample_signal_temp,num_upsamples)    
    
    # Delay impulse (But keep start after 37 to allow all delays to run)
    random_delay = int(np.random.uniform()*16)+37

    upsample_signal = np.roll(upsample_signal_temp,random_delay*upsample)
    #old_upsample_signal = np.roll(old_upsample_signal_temp,random_delay*upsample)
    #upsample_signal[random_delay*upsample:num_upsamples] = upsample_signal_temp[0:num_upsamples-random_delay*upsample]
    
    sample_period = upsample_time[1]-upsample_time[0]
    phase = int(np.random.uniform()*upsample)
    
    for t in range(0,num_samples):
        signal[t] = upsample_signal[t*upsample+phase]

    # Plot the frequency response of the antenna
    
    if(draw_flag):
        plt.figure(1,figsize=(8,4))
        plt.clf()
        plt.semilogy(((freq_response * 0.5 / np.pi) * w)/(10**9), abs(h), label="order = %d" % order)
        plt.axis([0,1.4,0.01,10])
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Gain')
        plt.title('Simulated Bandpass Filter')
        plt.grid(True)
    
    if (draw_flag==1):
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
    
    if (draw_flag==1):
        plt.figure(4,figsize=(8,4))
        plt.plot(time[0:num_samples],1000.0*signal[0:num_samples])
        plt.axis([0,35,-1500,1500])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("2.6 GSa/sec Impulse Simulation")
        plt.show()

    return signal


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    sample_frequency = 2600000000.0
    sample_length = 80
    impulse_position = 10
    upsample = 10
    draw_flag = 1
    
    upsample_length = upsample*sample_length
    
    impulse_sample = np.zeros(sample_length)
    impulse_upsample_fit = np.zeros(upsample_length)
    
    time = np.linspace(0.0,((sample_length*(10**9))/sample_frequency), sample_length)
    impulse_sample = impulse_gen(sample_length,impulse_position,upsample,sample_frequency,draw_flag=draw_flag)
    
    plt.figure(1)
    plt.clf()
    plt.plot(time,impulse_sample[0:sample_length])
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title("Impulse Simulation")
    plt.show()
