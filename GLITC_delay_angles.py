#!/usr/bin/env python

def calculate_angles():
	
	import numpy as np  
    
	speed_of_light = 2.99*10**8
	sample_period = 1.0/2600000000.0
	
	#bc_distance = 1.0861
	
	ab_top_horz_distance = 1.07721
	ab_top_vert_distance = 3.66827
	ab_bot_horz_distance = 1.30454
	ab_bot_vert_distance = 2.69545
	
	ac_top_horz_distance = 1.07721
	ac_top_vert_distance = 4.75437
	ac_bot_horz_distance = 1.30454
	ac_bot_vert_distance = 3.78155
	
	ab_top_distance = np.sqrt((ab_top_horz_distance)**2+(ab_top_vert_distance)**2)
	ab_bot_distance = np.sqrt((ab_bot_horz_distance)**2+(ab_bot_vert_distance)**2)
	ac_top_distance = np.sqrt((ac_top_horz_distance)**2+(ac_top_vert_distance)**2)
	ac_bot_distance = np.sqrt((ac_bot_horz_distance)**2+(ac_bot_vert_distance)**2)
	#num_angles = len(GLITC_delays)
	#print "Number of angles: " +str(num_angles)
	#print "Distance from antenna C to B: "+str(bc_distance)
	print "Distance from antenna C to top A: "+str(ab_top_distance)
	print "Distance from antenna C to bot A: "+str(ab_bot_distance)
	print "Distance from antenna C to top A: "+str(ac_top_distance)
	print "Distance from antenna C to bot A: "+str(ac_bot_distance)
	"""
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
		
	"""
	#bc_delay = np.zeros(100)
	ab_top_delay = np.zeros(100)
	ab_bot_delay = np.zeros(100)
	ac_top_delay = np.zeros(100)
	ac_bot_delay = np.zeros(100)
	counter = 0
	
	top_outfile = open('GLITC_upper_delays.dat','w')
	top_outfile.write('A_upper_delay,\tA_upper-B_delay,\tA_upper-C_delay\n')
	
	bot_outfile = open('GLITC_lower_delays.dat','w')
	bot_outfile.write('A_lower_delay,\tA_lower-B_delay,\tA_lower-C_delay\n')
	
	lowest_angle = -10
	highest_angle = 50
	angle = np.linspace(lowest_angle,highest_angle,100000)
	#print angle
	
	
	for i in range(0,len(angle)):
		unique_delay = True
		
		j=0
		#bc_delay[counter] = int((bc_distance*np.sin(angle[i]*(np.pi/180)))/(speed_of_light*sample_period))
		ab_top_delay[counter] = (-1.0)*int((ab_top_distance*np.cos(np.arctan(ab_top_vert_distance/ab_top_horz_distance)-(angle[i]*(np.pi/180.0))))/(speed_of_light*sample_period))
		ac_top_delay[counter] = (-1.0)*int((ac_top_distance*np.cos(np.arctan(ac_top_vert_distance/ac_top_horz_distance)-(angle[i]*(np.pi/180.0))))/(speed_of_light*sample_period))
		#print "\nFor angle: "+str(-1*angle[i])+ " degrees,  the  A_top_B_delay  should  be "+str(ab_top_delay[counter])
		#print "For angle: "+str(-1*angle[i])+ " degrees, the A_top_C_delay should be "+str(ac_top_delay[counter])
		
		# Check for duplicates
		for j in range(0,counter):
			#print 'checking for duplicate'
			if ((ab_top_delay[j] == ab_top_delay[counter]) and ((ac_top_delay[j] == ac_top_delay[counter]))):
				#print "BC and TOP not unique"
				unique_delay = False
		# If the delay is unique, save it
		#print "Save Delays"
		if (unique_delay):
			#print 'writing top delay'
			top_outfile.write('0,\t'+str(ab_top_delay[counter])+',\t'+str(ac_top_delay[counter])+'\n')
			counter +=1 
		#print counter
	
	counter=0
	for i in range(0,len(angle)):
		unique_delay = True
		j=0
		#bc_delay[counter] = int((bc_distance*np.sin(angle[i]*(np.pi/180)))/(speed_of_light*sample_period))
		ab_bot_delay[counter] = (-1.0)*int((ab_bot_distance*np.cos(np.arctan(ab_bot_vert_distance/ab_bot_horz_distance)-(angle[i]*(np.pi/180.0))))/(speed_of_light*sample_period))
		ac_bot_delay[counter] = (-1.0)*int((ac_bot_distance*np.cos(np.arctan(ac_bot_vert_distance/ac_bot_horz_distance)-(angle[i]*(np.pi/180.0))))/(speed_of_light*sample_period))
		#print "\nFor angle: "+str(-1*angle[i])+ " degrees,  the  A_bot_B_delay  should  be "+str(ab_bot_delay[counter])
		#print "For angle: "+str(-1*angle[i])+ " degrees, the A_bot_C_delay should be "+str(ac_bot_delay[counter])
		
		# Check for duplicates
		for j in range(0,counter):
			#print 'checking for duplicate'
			if((ab_bot_delay[j] == ab_bot_delay[counter]) and ((ac_bot_delay[j] == ac_bot_delay[counter]))):
				#print "BC and Bot not unique"
				unique_delay = False
		# If the delay is unique, save it
		#print "Save Delays"
		if (unique_delay):
			#print 'writing bot delay'
			bot_outfile.write('0,\t'+str(ab_bot_delay[counter])+',\t'+str(ac_bot_delay[counter])+'\n')
			counter +=1 
		#print counter
		
		
	top_outfile.close()
	bot_outfile.close()
	return 0
	
if __name__ == '__main__':
	calculate_angles()

	
                          
