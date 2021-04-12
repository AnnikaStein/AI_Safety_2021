#!/usr/bin/python3
## file: submit_auc.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os
import subprocess


import argparse


parser = argparse.ArgumentParser(description="Setup for AUC for custom tagger outputs")
parser.add_argument('-i',"--files", type=int, help="Number of files", default=1)
parser.add_argument('-m',"--mode", type=str, help="Mode: raw, noise, FGSM", default="raw")
args = parser.parse_args()

NUM_DATASETS = args.files
mode = args.mode

home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/new_march_21/code/"
print('Shell script is located at:\t',shPath)

if mode == 'raw':
    time = '01'
    params = ['0.0']
elif mode == 'noise':
    time = '02'
    params = ['0.005','0.01','0.05','0.1']
elif mode == 'FGSM':
    time = '05'
    params = ['0.005','0.01','0.05','0.1']

for param in params:
    
    submit_command = ("sbatch "
            "--time={3}:00:00 "
            "--job-name=AUCOutputs_{0}_{1}_{2} "
            "--export=NUM_F={0},MODE={1},PARAM={2} {4}auc_outputs_tt_attack.sh").format(NUM_DATASETS, mode, param, time, shPath)
    
    #print(submit_command)
    exit_status = subprocess.call(submit_command, shell=True)
    if exit_status==1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(submit_command))
print("Done submitting jobs!")