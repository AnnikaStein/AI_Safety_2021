#!/usr/bin/python3
## file: submit_compare_weighted.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os
import subprocess


import argparse


parser = argparse.ArgumentParser(description="Setup for AUC for custom tagger outputs")
parser.add_argument('-f',"--fromVar", type=int, help="Starting number input variable", default=0)
parser.add_argument('-t',"--toVar", type=int, help="End with this input variable", default=66)
parser.add_argument('-m',"--mode", type=str, help="Mode: noise, fgsm", default="noise")
parser.add_argument('-r',"--fixRange", help="Use predefined range (yes) or just as is (no)", default='yes')
parser.add_argument('-ed',"--evaldataset", type=str, help="Dataset used during evaluation, qcd or tt", default="tt")
parser.add_argument('-td',"--traindataset", type=str, help="Dataset used during training, qcd or tt", default="tt")
args = parser.parse_args()

fromVar = args.fromVar
toVar = args.toVar
mode = args.mode
fixRange = args.fixRange
evaldataset = args.evaldataset
traindataset = args.traindataset

home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/"
print('Shell script is located at:\t',shPath)

if mode == 'noise':
    time = '12'
elif mode == 'fgsm':
    time = '15'

    
if fromVar == 0 and toVar == 66:
    indexlist = [[0, 11], [12, 21], [22, 34], [35, 48], [49, 58], [59, 66]]
    
else:
    indexlist = [[fromVar, toVar]]
    
for indices in indexlist:
    
    submit_command = ("sbatch "
            "--time={6}:00:00 "
            "--job-name=AUCOutputs_{0}_{1}_{2}_{3}_{4}_{5} "
            "--export=FROM={0},TO={1},MODE={2},FIXRANGE={3},EVALDATASET={4},TRAINDATASET={5} {7}compare_weighted.sh").format(indices[0], indices[1], mode, fixRange, evaldataset, traindataset, time, shPath)
    
    print(submit_command)
    exit_status = subprocess.call(submit_command, shell=True)
    if exit_status==1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(submit_command))
print("Done submitting jobs!")