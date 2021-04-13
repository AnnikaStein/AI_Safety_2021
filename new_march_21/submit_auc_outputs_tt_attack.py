#!/usr/bin/python3
## file: submit_auc_outputs_tt_attack.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os
import subprocess


import argparse


parser = argparse.ArgumentParser(description="Setup for AUC for custom tagger outputs")
parser.add_argument('-i',"--files", type=int, help="Number of files", default=1)
parser.add_argument('-m',"--mode", type=str, help="Mode: raw, deepcsv, noise, FGSM", default="raw")
parser.add_argument('-td',"--traindataset", type=str, help="Dataset used during training, qcd or tt", default="tt")
args = parser.parse_args()

NUM_DATASETS = args.files
mode = args.mode
traindataset = args.traindataset

home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/new_march_21/code/"
print('Shell script is located at:\t',shPath)

if mode == 'raw' or mode == 'deepcsv':
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
            "--time={4}:00:00 "
            "--job-name=AUCOutputs_{0}_{1}_{2}_{3} "
            "--export=NUM_F={0},MODE={1},PARAM={2},TRAINDATASET={3} {5}auc_outputs_tt_attack.sh").format(NUM_DATASETS, mode, param, traindataset, time, shPath)
    
    print(submit_command)
    exit_status = subprocess.call(submit_command, shell=True)
    if exit_status==1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(submit_command))
print("Done submitting jobs!")
