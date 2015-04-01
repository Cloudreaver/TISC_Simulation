#!/usr/bin/env python

def calculate_angles():
	
	import numpy as np
	
	GLITC_delays = np.array([[0,30,42],
                           [0,29,41],
                           [0,29,40],
                           [0,28,38],
                           [0,27,37],
                           [0,27,36],
                           [0,26,35],
                           [0,25,34],
                           [0,25,33],
                           [0,24,32],
                           [0,23,31],
                           [0,23,30],
                           [0,22,29],
                           [0,21,27],
                           [0,20,26],
                           [0,20,25],
                           [0,19,24],
                           [0,18,23],
                           [0,18,22],
                           [0,17,21],
                           [0,17,20],
                           [0,16,20],
                           [0,16,19],
                           [0,15,18],
                           [0,15,17],
                           [0,14,16],
                           [0,13,15],
                           [0,13,14],
                           [0,12,13],
                           [0,12,12],
                           [0,11,11],
                           [0,10,10],
                           [0,9,9],
                           [0,8,8],
                           [0,8,7],
                           [0,8,6],
                           [0,7,5],
                           [0,6,4],
                           [0,6,3],
                           [0,5,3],
                           [0,5,2],
                           [0,4,1],
                           [0,0,0],
                           [16,19,15],
                           [16,18,14],
                           [16,18,13],
                           [16,17,12],
                           [16,17,11],
                           [16,16,11],
                           [16,15,10],
                           [16,15,9],
                           [16,14,8],
                           [16,14,7],
                           [16,13,12],
                           [16,12,5]])
	
    
    
	speed_of_light = 2.99*10**8
	sample_period = 3.84*10**(-10)
	ab_distance = 1.0861
	ac_top_horz_distance = 1.07721
	ac_top_vert_distance = 4.75437
	ac_bot_horz_distance = 1.30454
	ac_bot_vert_distance = 3.78155
	ac_top_distance = np.sqrt((ac_top_horz_distance)**2+(ac_top_vert_distance)**2)
	ac_bot_distance = np.sqrt((ac_bot_horz_distance)**2+(ac_bot_vert_distance)**2)
	num_angles = len(GLITC_delays)
	print "Number of angles: " +str(num_angles)
	print "Distance from antenna A to B: "+str(ab_distance)
	print "Distance from antenna A to top C: "+str(ac_top_distance)
	print "Distance from antenna A to bot C: "+str(ac_bot_distance)
    
	for i in range(0,num_angles):
		
		ab_theta = np.arcsin((speed_of_light*(GLITC_delays[i][1]-GLITC_delays[i][0])*sample_period)/ab_distance)
		#print ab_distance
		ab_theta *= 180/np.pi
		ac_top_theta = np.arctan(ac_top_vert_distance/ac_top_horz_distance)-np.arcsin((speed_of_light*(GLITC_delays[i][2]-GLITC_delays[i][0])*sample_period)/ac_top_distance)
		ac_top_theta*= 180/np.pi
		ac_bot_theta = np.arctan(ac_bot_vert_distance/ac_bot_horz_distance)-np.arcsin((speed_of_light*(GLITC_delays[i][2]-GLITC_delays[i][0])*sample_period)/ac_bot_distance)
		ac_bot_theta*= 180/np.pi
		
		print "\nGLITC delay of "+str(GLITC_delays[i][1])+" means AB Angle: " +str(ab_theta)
		print "GLITC delay of "+str(GLITC_delays[i][2])+" means AC Top Angle: "+str(ac_top_theta)
		print "GLITC delay of "+str(GLITC_delays[i][2])+" means AC Bot Angle: "+str(ac_bot_theta)
		
		
	for i in range(-10,90):
			ab_delay = int((ab_distance*np.sin(i*(np.pi/180)))/(speed_of_light*sample_period))
			print "For angle: "+str(i)+ " degrees, the AB_delay should be "+str(ab_delay)
	
	
if __name__ == '__main__':
	calculate_angles()

	
                          
