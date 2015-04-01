#!/bin/bash

#PBS -l walltime=40:00:00
#PBS -l nodes=1:ppn=1
#PBS -m e
#PBS -M hupe.2@buckeyemail.osu.edu
#PBS -o output/PBS_output
#PBS -j oe

#OPTIONS FOR PBS

cd /home/hupe/thesis_code #probably not necessary, precaution
source /home/hupe/.bash_profile


./run_TISC_sim_at_rate.py #executable I want to run