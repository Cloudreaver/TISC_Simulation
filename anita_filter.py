#!/usr/bin/env python


from scipy.signal import butter, lfilter,lfiltic
import csv
import numpy as np
from scipy.fftpack import fft, ifft

#import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs=2600000000.0, order=2):
    nyq = 0.5 * fs 
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut=250000000.0, highcut=900000000.0, fs=2600000000.0, order=2):
    from noise import generate_noise
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = generate_noise(len(a),noise_sigma=32.0,filter_flag=0)
    x = np.linspace(0.0,(1.0/fs)*len(data),len(b))
    zi = lfiltic(b,a,y,x)
    y,zf = lfilter(b,a,data,zi=zi)
    return y

"""
# Figure this out later

def anita_filter():
	
	time_list = []
	signal_list = []

	counter = 0

	with open('anita_II_impulse_response.dat', 'rb') as csvfile:
		data = csv.reader(csvfile, delimiter=' ',quotechar='|')
		for row in data:
			time_list.append(float(row[0]))
			signal_list.append(float(row[1]))
			counter +=1
			
	time = np.array(time_list)
	signal = np.array(signal_list)
	


	plt.figure(1)
	plt.plot(time,signal)
	plt.xlabel("Time [ns]")
	plt.ylabel("Voltage [mV]")
	plt.title("ANITA II Impulse Response")



	num_samples = len(time)
	T = time[1]-time[0]

	freq_response = np.fft.rfft(signal)
	freq_response = freq_response/(0.008*np.amax(freq_response))
	frequency = np.linspace(0.0,1.0/(2.0*T),num_samples/2)

	#print freq_response

	plt.figure(2)
	plt.clf()
	plt.semilogy(frequency,2.0/num_samples * np.abs(freq_response[0:num_samples/2]))
	#plt.axis([0,1.5,0,1.1])
	plt.xlabel('Frequency (MHz)')
	plt.ylabel('Gain')
	plt.title('Antenna Impulse Reponse')
	plt.grid(True)
	plt.show()
	"""

if __name__ == '__main__':
	print "test"
	
	
	
