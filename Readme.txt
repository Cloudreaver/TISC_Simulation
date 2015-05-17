Repository Name: TISC_Simulation
Author Name: Ryan Hupe

SYNOPSYS
This compilaiton of code simulations the Triggering Interferometric Sum Correlator (TISC) board, planned for use in the fourth flight of the Antarctic Impulsive Transient Antenna (ANITA). 

PREREQUISITES
Python,Numpy, Scipy, Matplotlib, ROOT(for now)

RUNNING THE SIMULATION
There are currently ways of running a full simulation of the code. 

run_TISC_sim_at_rate.py: This file will run over a specified number of impulsive events at a specified event and simulation rate. CW and thermal noise levels can be adjusted as well as the SNR of the impulsive signal. Most parameters are commented in the code. At the end of the code, a plot of all the Triggering Rate vs threshold level should be plotted for each SNR.

run_avg_subtracted_TISC_sim: This file is nearly identical to the previous one, with the exception of using a new average subtracted triggering algorithm. This algorithm runs a specified number of trial events and keeps a running average for each correlation. Onces these trial events are finished. The code will simulate a specified number of events at a event/simulation rate and will subtract the calculated average from each correlation for each event.

run_3_phi_TISC_sim_at_rate.py: WORK IN PROGRESS
  
MODULE DESCRIPTIONS
All modules should be able to be run on their own and should plot their results. It should give the user an understanding of hos they work. All of the modules inputs and outputs should have fairly self explanitory names and should be commented.

GLITC_delay_angles.py: This file calculates the signal delays for various incident angles. All delays are with respect to antenna A (top ring of antennas). The delays are calculated two seperate times, one for the upper A antennas and another for the lower A antennas. Antenna distances were calculated from positions specified by C. Miki of University of Hawaii.

TISC_sim.py: This module simulates a single event and the TISC response to that event. It uses several sub-modules that will be described in full later in this readme. First, gaussian thermal noise is generated for each of the three antennas via a noise.py module. Then a CW signal is generated via cw.py module, if desired, and added to the thermal noise. This CW is phase aligned for each of the three channels, so the same CW signal is used for all three antennas. This CW+thermal noise is then filtered via filter.py modules, which passes the noise through an approximation of the ANITA impulse response. Then an impulsive signal is created by the impulse.py module and amplified to achieve the desired signal to noise ratio. These impulses are then added to the noise, resulting in waveforms similar to what the TISC would recieve in flight. These waveforms are then digitized via the digitizer.py module. The digitized waveforms are then feed into the correlator.py module which adds the three wavefrom element wise are various delays and returns the maximum sum out of all the delays as well as a trigger decision based on a given threshold.

noise.py: This module generates a waveform of given length filled with random gaussian noise of a given mean and sigma. It also has an optional filter flag that will pass the noise through the simulation anita impulse response.

cw.py: This module generates a waveform of a given length filled with an amplitude modulated carrier wave signal with given carrier and modulation frequencies and amplitude. It also has the option to generate a random phase offset.

filter.py: This module takes in a waveform and passes if through the simulated anita impulse response. The impulse response is simulated by a butterworth bandpass filter of order 2 with corner(-3dB) frequencies at 250MHz and 900MHz with a sample frequency of 2.6GHz.

impulse.py: This module generates an impulse waveform of given length by taking the anita impulse response and taking an inverse fft. Since the incoming impulse signal is simply a delta function in time and flat in frequency, it would simply map out the impulse reponse of the instrument.

digitizer.py: This module takes in a given waveform and digitizes it into N-bit values (default is 3-bit) using a given digitization_factor as the theshold inverval (default if 20mV). The possible output values are centered around zero.

sum_correlator.py: This module take in three digitized waveforms and performs a sum correlation on them at various delays corresponding to different angles(see GLITC_delay_angles.py). It has two options, one to use only those delays that are specified and the other is to run over all possible delays. It also has the option to use the average subtracted algorithm. If the average subtracted algorithm is used, you must specify the trail number that you are using to calculate the averages correctly. Once the trials are done, set the trial number to zero to stop calculating new averages and just use the existing one.

make_plots.py: This module just takes all the individual modules and plots their results for the given parameters.

batch_loops.sh/avg_sub_batch_loops.sh: These are used to call TISC_sim.sh/avg_sub_TISC_sim.sh for various parameters and submit them to the Kingbee cluster. This won't work on any other system other than kingbee at OSU

TISC_sim.sh/avg_sub_TISC_sim.sh: These scripts actually do the submitting to the cluster. Again, this will only work on kingbee at OSU.

three_phi_sector*: These are a version of TISC_sim that calculates everything for 3 phi sectors instead of just 1. It is still a work in progress.



TO DO:

Fix the plotting of the SNR curves in TISC_sim.py

Get the average subtracted algorithm working correctly.

Swap antenna A and C. Currently I have the bottom antenna being A and the top being C, this is backwards. This will require some thinking because the way I handle the correlations will probalby have to change.

Get the 3 phi sector version working. This shouldn't be hard, since it is just doing the same thing three times.
