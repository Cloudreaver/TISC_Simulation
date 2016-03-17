#!/usr/bin/env python



class sim_param:
	# TISC settings parameters
   TISC_sample_length = 16
   sample_frequency = 2600000000.0
   num_bits = 3
   digitization_factor=32.0
   window_length = 8
   window_weight = 0.5
   num_delays = [63,46]
   
   # Impulse parameters
   SNR = 5
   num_samples = 80
   impulse_position = 37
   b_input_delay = -15 # These delays should be negative, since A is the 
   c_input_delay = -17 # top antenna and we expect an upgoing signal
   boresight=0
   baseline = 0
   upsample = 10
   impulse_type = 0 # 0=Testbench impulse, 1 = ANITA I impulse
   filter_order = 8
   
   # Thermal & CW noise parameters
   noise_sigma = 32.0
   cw_flag = False
   cw_amplitude = 1.0*noise_sigma
   cw_frequency = 260000000.0
   modulation_frequency = 1.0
   
   # Simulation Control
   num_events=10000
   angle_range = 100
   simulation_rate = 162500000.0          # Simulation Rate
   event_rate = 81250000.0                 # Rate to generate events
   low_SNR = 0.0                   # Lowest SNR
   high_SNR = 5.0                 # Highest SNR
   step_SNR = 1.0                # SNR interval
   low_threshold = 0             # Lowest Threshold
   high_threshold = 4095        # Highest Threshold
   step_threshold = 1            # Threshold Interval
   draw_flag = False
   debug = False
   delay_type_flag = 1
   average_subtract_flag = False
   six_phi_sector_add = False
   SNR_draw_flag = True             # 1=draw SNR graphs, 0=don't
   
   # Output control
   output_dir = "output/"
