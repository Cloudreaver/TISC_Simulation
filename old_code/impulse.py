#!/usr/bin/env python

from scipy.signal import butter, lfilter, interpolate, resample
from numpy import tile, newaxis, dot, sinc

def butter_bandpass(lowcut, highcut, fs=2800000000.0, order=4):
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band',analog=False)
    return b, a


def butter_bandpass_filter(data, lowcut=132000000.0, highcut=1300000000.0, fs=2800000000.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def sinc_interp(x,s,u):
    """
    Interpolates x, sampled at "s"  instants
    Output y is sampled at "u" instants ("u" for "upsampled")

    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html
    """
    
    if len(x) != len(s):
        raise Exception, 'x and s must be the same length'
    
    # Find the period
    T = s[1]-s[0]
    
    sincM = tile(u, (len(s),1)) - tile(s[:,newaxis], (1,len(u)))
    y = dot(x,sinc(sincM/T))
    return y


def impulse_gen(num_samples,delay,upsample=1,freq_response=2800000000.0,low_corner_freq=132000000.0,high_corner_freq=1300000000.0,order=4,draw_flag=0,output_dir='output/'):
    # Input defaults are set for TISC bandpass filters

    import numpy as np
    if(draw_flag):
        import matplotlib.pyplot as plt
    from scipy.signal import freqz
    from scipy import ifft, shape

    num_upsamples = num_samples * upsample
    delay *= upsample
    nyquist_freq = freq_response/2
    #freq_response *= upsample
    # Plot the frequency response of the filter
    if(draw_flag):
        plt.figure(1)
        plt.clf()

    b, a = butter_bandpass(low_corner_freq, high_corner_freq, order=order)
    w, h = freqz(b, a, worN=num_samples)

    if(draw_flag):
        plt.plot((freq_response * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        #plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],'--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Bandpass filter')
        plt.grid(True)
        #plt.legend(loc='best')
    
    # Transform bandpass filter into time domain to get impulse signal
    #print len(h)
    #print num_upsamples
    signal_temp = ifft(h,num_upsamples)
    #print len(signal_temp)
    # Delay impulse 
    signal = np.zeros(num_upsamples)
    signal[delay:num_upsamples] = signal_temp[0:num_upsamples-delay]
    
    # Upsample
    time = np.linspace(0.0,(num_upsamples/(2400000000.0*upsample))*(10**9), num_upsamples)
    #upsample_time = np.linspace(0.0,((num_upsamples)/(2400000000.0*upsample))*(10**9), num_upsamples)
    #upsample_signal = resample(signal, num_upsamples)
    upsample_time = time
    upsample_signal = signal
    #upsample_signal = np.interp(upsample_time,time,signal)    
    #upsample_signal = sinc_interp(signal,time,upsample_time)
    

    if (draw_flag==1):
        #time = np.linspace(0.0,((num_upsamples)/(2400000000.0*upsample))*(10**9), num_upsamples)
        plt.figure(2)
        plt.plot(upsample_time,upsample_signal[0:num_upsamples])
        plt.xlabel("Time [ns]")
        plt.ylabel("Amplitude")
        plt.title("Impulse Signal")
        #plt.legend(loc='best')
        plt.show()
    #print len(upsample_signal)
    #print upsample_signal
    return upsample_signal


if __name__ == '__main__':
    impulse_gen(512,10,upsample=32,draw_flag=1,output_dir='output/')
