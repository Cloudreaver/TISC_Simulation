#!/usr/bin/env python

from scipy.signal import butter, filtfilt, lfilter, resample


def butter_bandpass(lowcut, highcut, fs=2800000000.0, order=2):
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut=132000000.0, highcut=1300000000.0, fs=2800000000.0, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    #y = filtfilt(b, a, data)
    y = lfilter(b,a,data)
    return y
    
def sinc_interp(x, s, u):
    import numpy as np
    
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")
    
    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html        
    """
    
    if len(x) != len(s):
        raise Exception, 'x and s must be the same length'
    
    
    # Find the period    
    T = s[1] - s[0]
    #print T
    sincM = np.tile(u, (len(s), 1)) - np.tile(s[:, np.newaxis], (1, len(u)))
    #print sincM
    y = np.dot(x, np.sinc(sincM/T))
    return y




def impulse_gen(num_samples,delay,upsample=10,freq_response=2600000000.0,low_corner_freq=132000000.0,high_corner_freq=900000000.0,order=2,draw_flag=0,output_dir='output/'):
    # Input defaults are set for TISC bandpass filters

    import numpy as np
    if(draw_flag):
        import matplotlib.pyplot as plt
    from scipy.signal import freqz
    from scipy import ifft, shape
    from scipy.interpolate import interp1d

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

    b, a = butter_bandpass(low_corner_freq, high_corner_freq, order=4,fs=2600000000.0)
    w, h = freqz(b, a, worN=num_samples)
    
    # Plot the frequency response of the antenna
    if(draw_flag):
        plt.figure(1)
        plt.clf()
        plt.plot(((freq_response * 0.5 / np.pi) * w)/(10**9), abs(h), label="order = %d" % order)
        plt.axis([0,1.5,0,1.1])
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Gain')
        plt.title('Antenna Impulse Reponse')
        plt.grid(True)
        
    # Transform bandpass filter into time domain to get impulse signal
    upsample_signal_temp = ifft(h,num_upsamples).real
        
    # Delay impulse 
    random_delay = int(np.random.uniform()*16)

    upsample_signal[random_delay*upsample:num_upsamples] = upsample_signal_temp[0:num_upsamples-random_delay*upsample]
    
    if (draw_flag==1):
        plt.figure(2)
        plt.plot(upsample_time,1000.0*upsample_signal[0:num_upsamples])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("Upsampled Impulse Simulation")
        #plt.show()

    
    sample_period = upsample_time[1]-upsample_time[0]
    phase = int(np.random.uniform()*upsample)
    
    for t in range(0,num_samples):
        signal[t] = upsample_signal[t*upsample+phase]

    
    if (draw_flag==1):
        plt.figure(3)
        plt.plot(time,1000.0*signal[0:num_samples])
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("Impulse Simulation")
        plt.show()
    #print len(signal)
    return signal#, upsample_signal


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from noise import generate_noise
    from digitizer import digitize
    from cw import generate_cw
    
    sample_frequency = 2600000000.0
    sample_length = 74
    impulse_position = 10
    upsample = 10
    draw_flag = 0
    output_dir = 'output/'
    clock_jitter = 0.000000001
    #poly_index = 5
    SNR = 3
    noise_sigma = 20
    save_plot_flag = False
    cw_flag = False
    carrier_frequency=260000000.0    # Hz
    modulation_frequency=16000000.0  # Hz
    peak_amplitude = 0.5*noise_sigma #Peak amplitude in mV
    
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
    impulse_sample, impulse_upsample = impulse_gen(sample_length,impulse_position,upsample,sample_frequency,draw_flag=draw_flag,output_dir=output_dir)
    """
    plt.figure(1)
    plt.clf()
    plt.plot(time,impulse_sample[0:sample_length])
    plt.xlabel("Time [ns]")
    plt.ylabel("Amplitude [unitless]")
    plt.title("Impulse Simulation")
    if (save_plot_flag):
        plt.savefig("plots/impulse_sample.png")
    
    upsample_time = np.linspace(0.0,((upsample_length*(10**9))/(sample_frequency*upsample)), upsample_length)
    #impulse_upsample = impulse_gen(sample_length,impulse_position,upsample,sample_frequency,draw_flag=draw_flag,output_dir=output_dir)
    plt.figure(2)
    plt.clf()
    plt.plot(upsample_time,impulse_upsample[0:upsample_length])
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title("Upsampled Impulse Simulation")
    if (save_plot_flag):
        plt.savefig("plots/upsample_impulse_sample.png")
      
    signal_amp = (2*SNR*noise_sigma)
    difference=np.amax(impulse_sample)-np.amin(impulse_sample) # Get peak to peak voltage
    impulse_sample *= (1/difference) # Normalize input
    impulse_sample *= signal_amp # Amplify
    
    noise_sample = generate_noise(sample_length=sample_length,upsample=1,noise_mean=0.0,noise_sigma=noise_sigma,filter_flag=1)
    plt.figure(3)
    plt.clf()
    plt.plot(time,noise_sample)
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title("Band Limited Noise Sample")
    if (save_plot_flag):
        plt.savefig("plots/noise_sample.png")
    
    if (cw_flag):
        cw_sample = generate_cw(sample_length=sample_length,upsample=1,sample_frequency=sample_frequency,carrier_frequency=carrier_frequency,modulation_frequency=modulation_frequency,peak_amplitude=peak_amplitude,filter_flag=1)
        plt.figure(4)
        plt.clf()
        plt.plot(time,cw_sample)
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("Carrier Wave Sample (Peak Amp: "+str(peak_amplitude/noise_sigma)+" x Noise RMS)")
        if (save_plot_flag):
            plt.savefig("plots/cw_sample.png")
        
        noise_sample = np.add(noise_sample,cw_sample)
        plt.figure(5)
        plt.clf()
        plt.plot(time,noise_sample)
        plt.xlabel("Time [ns]")
        plt.ylabel("Voltage [mV]")
        plt.title("Noise+CW Sample")
        if (save_plot_flag):
            plt.savefig("plots/noise_cw_sample.png")
        
    impulse_noise_sample = np.add(impulse_sample,noise_sample)
    plt.figure(6)
    plt.clf()
    plt.plot(time,impulse_noise_sample)
    plt.xlabel("Time [ns]")
    plt.ylabel("Voltage [mV]")
    plt.title("Impulse(SNR "+str(SNR)+") + Noise Sample(RMS "+str(noise_sigma)+"mV)")
    if (save_plot_flag):
        plt.savefig("plots/impulse_noise_sample.png")
    
    
    digitized_sample = digitize(impulse_noise_sample,sample_length,upsample=1,num_bits=3,noise_mean=0.0,noise_rms=20,digitization_factor=1)
    plt.figure(8)
    plt.clf()
    plt.plot(time,digitized_sample)
    plt.xlabel("Time [ns]")
    plt.ylabel("Digital Levels [unitless]")
    plt.title("Digitized Impulse + Noise Sample")
    if (save_plot_flag):
        plt.savefig("plots/digitized_impulse_noise_sample.png")
    
    #for index in range(0,sample_length):
        #if (upsample > 1):
            #downsample_random = random.randrange(0,1)
            #downsample_random = random.randrange(0,upsample-1)
        #else:
            #downsample_random = 0
            
        #impulse_downsample[index] = impulse_upsample[(index*upsample)+downsample_random]
    
    #print impulse_downsample
    #plt.figure(3)
    #plt.clf()
    #plt.plot(time,impulse_downsample[0:sample_length])
    #plt.xlabel("Time [ns]")
    #plt.ylabel("Voltage [mV]")
    #plt.title("Downsampled Impulse Simulation")
    """
    
    plt.show()
