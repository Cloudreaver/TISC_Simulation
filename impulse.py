#!/usr/bin/env python

from scipy.signal import butter, filtfilt


def butter_bandpass(lowcut, highcut, fs=2800000000.0, order=4):
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')#,analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut=132000000.0, highcut=1300000000.0, fs=2800000000.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y




def impulse_gen(num_samples,delay,upsample=1,freq_response=2800000000.0,low_corner_freq=132000000.0,high_corner_freq=1300000000.0,order=4,draw_flag=0,output_dir='output/'):
    # Input defaults are set for TISC bandpass filters

    import numpy as np
    if(draw_flag):
        import matplotlib.pyplot as plt
    from scipy.signal import freqz
    from scipy import ifft, shape

    # Get number of samples in upscaled signal
    num_upsamples = num_samples*upsample

    # Adjust the delay for upsampling
    delay *= upsample
    nyquist_freq = freq_response/2
    
    # Plot the frequency response of the filter
    if(draw_flag):
        plt.figure(1)
        plt.clf()

    b, a = butter_bandpass(low_corner_freq, high_corner_freq, order=order)
    w, h = freqz(b, a, worN=num_samples)

    if(draw_flag):
        plt.plot(((freq_response * 0.5 / np.pi) * w)/(10**9), abs(h), label="order = %d" % order)
        plt.axis([0,1.5,0,1.1])
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Gain')
        plt.title('Butterworth Bandpass Filter')
        plt.grid(True)
        
    
    # Transform bandpass filter into time domain to get impulse signal
    signal_temp = ifft(h,num_upsamples)
    
    # Delay impulse 
    signal = np.zeros(num_upsamples)
    signal[delay:num_upsamples] = signal_temp[0:num_upsamples-delay]
    
    # Get time axis
    time = np.linspace(0.0,(num_samples/freq_response)*(10**9), num_upsamples)

    if (draw_flag==1):
        plt.figure(2)
        plt.plot(time,1000*signal[0:num_upsamples])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("Impulse Simulation")
        plt.show()
    return signal


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from noise import generate_noise
    from digitizer import digitize
    
    sample_frequency = 2800000000
    sample_length = 74
    impulse_position = 35
    upsample = 8
    draw_flag = 0
    output_dir = 'output/'
    clock_jitter = 0.000000001
    poly_index = 5
    SNR = 5
    noise_sigma = 100
    save_plot_flag = True
    
    upsample_length = upsample*sample_length
    
    impulse_sample = np.zeros(sample_length)
    impulse_upsample = np.zeros(upsample_length)
    impulse_downsample = np.zeros(sample_length)
    impulse_upsample_fit_poly = np.zeros(poly_index)
    impulse_upsample_fit = np.zeros(upsample_length)
    impulse_noise_sample = np.zeros(sample_length)
    digitized_sample = np.zeros(sample_length)
    
    time = np.linspace(0.0,((sample_length*(10**9))/sample_frequency), sample_length)
    impulse_sample = impulse_gen(sample_length,impulse_position,1,sample_frequency,draw_flag=draw_flag,output_dir=output_dir)
    plt.figure(1)
    plt.clf()
    plt.plot(time,impulse_sample[0:sample_length])
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title("Impulse Simulation")
    if (save_plot_flag):
        plt.savefig("impulse_sample.png")
    
    upsample_time = np.linspace(0.0,((upsample_length*(10**9))/sample_frequency), upsample_length)
    impulse_upsample = impulse_gen(sample_length,impulse_position,upsample,sample_frequency,draw_flag=draw_flag,output_dir=output_dir)
    plt.figure(2)
    plt.clf()
    plt.plot(upsample_time,impulse_upsample[0:upsample_length])
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title("Upsampled Impulse Simulation")
    if (save_plot_flag):
        plt.savefig("upsample_impulse_sample.png")
    
    signal_amp = (2*SNR*noise_sigma)
    difference=np.amax(impulse_sample)-np.amin(impulse_sample) # Get peak to peak voltage
    impulse_sample *= (1/difference) # Normalize input
    impulse_sample *= signal_amp # Amplify
    impulse_noise_sample = impulse_sample + generate_noise(sample_length=sample_length,upsample=1,noise_mean=380,noise_sigma=100,filter_flag=1)
    plt.figure(4)
    plt.clf()
    plt.plot(time,impulse_noise_sample)
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title("Impulse + Noise Sample")
    if (save_plot_flag):
        plt.savefig("impulse_noise_sample.png")
    
    
    digitized_sample = digitize(impulse_noise_sample,sample_length,upsample=1,num_bits=3,noise_mean=380,noise_rms=100,digitization_factor=1)
    plt.figure(5)
    plt.clf()
    plt.plot(time,digitized_sample)
    plt.xlabel("Time [ns]")
    plt.ylabel("Digital Levels [unitless]")
    plt.title("Digitized Impulse + Noise Sample")
    if (save_plot_flag):
        plt.savefig("digitized_impulse_noise_sample.png")
    
    for index in range(0,sample_length):
        if (upsample > 1):
            downsample_random = random.randrange(0,1)
            downsample_random = random.randrange(0,upsample-1)
        else:
            downsample_random = 0
            
        impulse_downsample[index] = impulse_upsample[(index*upsample)+downsample_random]
    
    #print impulse_downsample
    #plt.figure(3)
    #plt.clf()
    #plt.plot(time,impulse_downsample[0:sample_length])
    #plt.xlabel("Time [ns]")
    #plt.ylabel("Voltage [mV]")
    #plt.title("Downsampled Impulse Simulation")
    
    
    plt.show()
