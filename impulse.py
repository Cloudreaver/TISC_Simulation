#!/usr/bin/env python

def impulse_gen(num_samples,delay,upsample=10,freq_response=2600000000.0,low_corner_freq=250000000.0,high_corner_freq=900000000.0,order=2,draw_flag=0,output_dir='output/'):
    # Input defaults are set for TISC bandpass filters

    import numpy as np
    if(draw_flag):
        import matplotlib.pyplot as plt
    from scipy.signal import freqz
    from scipy import ifft, shape
    from scipy.interpolate import interp1d
    from anita_filter import butter_bandpass, butter_bandpass_filter

    # Get number of samples in upscaled signal
    num_upsamples = num_samples*upsample

    # Adjust the delay for upsampling
    nyquist_freq = freq_response/2
    
    signal = np.zeros(num_samples)
    signal_temp = np.zeros(num_samples)
    upsample_signal_temp = np.zeros(num_upsamples)
    upsample_signal = np.zeros(num_upsamples)
    
    # Get time axis
    time = np.linspace(0.0,(num_samples/freq_response)*(10**9), num_samples)
    upsample_time = np.linspace(0.0,(num_samples/freq_response)*(10**9), num_upsamples)

    b, a = butter_bandpass(low_corner_freq, high_corner_freq, order=order,fs=2600000000.0)
    w, h = freqz(b, a, worN=num_samples)
    
    # Transform bandpass filter into time domain to get impulse signal
    upsample_signal_temp = ifft(h,num_upsamples).real
        
    # Delay impulse (But keep start after 37 to allow all delays to run)
    random_delay = int(np.random.uniform()*16)+37

    upsample_signal[random_delay*upsample:num_upsamples] = upsample_signal_temp[0:num_upsamples-random_delay*upsample]
    
    sample_period = upsample_time[1]-upsample_time[0]
    phase = int(np.random.uniform()*upsample)
    
    for t in range(0,num_samples):
        signal[t] = upsample_signal[t*upsample+phase]

    # Plot the frequency response of the antenna
    if(draw_flag):
        plt.figure(1)
        plt.clf()
        plt.semilogy(((freq_response * 0.5 / np.pi) * w)/(10**9), abs(h), label="order = %d" % order)
        plt.axis([0,1.4,0.01,10])
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Gain')
        plt.title('Simulated Impulse Reponse')
        plt.grid(True)

    if (draw_flag==1):
        plt.figure(2)
        plt.plot(upsample_time,1000.0*upsample_signal[0:num_upsamples])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("Upsampled Impulse Simulation")
    
    if (draw_flag==1):
        plt.figure(3)
        plt.plot(time,1000.0*signal[0:num_samples])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("Impulse Simulation")
        plt.show()

    return signal


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    sample_frequency = 2600000000.0
    sample_length = 74
    impulse_position = 10
    upsample = 10
    draw_flag = 0
    
    upsample_length = upsample*sample_length
    
    impulse_sample = np.zeros(sample_length)
    impulse_upsample_fit = np.zeros(upsample_length)
    
    time = np.linspace(0.0,((sample_length*(10**9))/sample_frequency), sample_length)
    impulse_sample = impulse_gen(sample_length,impulse_position,upsample,sample_frequency,draw_flag=draw_flag)
    
    plt.figure(1)
    plt.clf()
    plt.plot(time,impulse_sample[0:sample_length])
    plt.xlabel("Time [ns]")
    plt.ylabel("Amplitude [unitless]")
    plt.title("Impulse Simulation")
    plt.show()
