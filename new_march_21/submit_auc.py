#!/usr/bin/python3
## file: submit_auc.py


# ================== Following / adapted from https://www.osc.edu/book/export/html/4046 ====================

import os
import subprocess


import argparse


parser = argparse.ArgumentParser(description="Setup for AUC ranking")
parser.add_argument('-i',"--files", type=int, help="Number of files for AUC ranking", default=1)
parser.add_argument('-s',"--start", type=int, help="Start index", default=0)
parser.add_argument('-e',"--end", type=int, help="End index", default=66)
parser.add_argument('-m',"--mode", type=str, help="Mode: raw, noise, FGSM", default="raw")
parser.add_argument('-p',"--param", type=float, help="Parameter used for attack, or 0 for raw", default=0.0)
parser.add_argument('-td',"--traindataset", type=str, help="Dataset used during training, qcd or tt", default="qcd")
args = parser.parse_args()

NUM_DATASETS = args.files
start = args.start
end = args.end
mode = args.mode
param = args.param
traindataset = args.traindataset

home = os.path.expanduser('~')
logPath = home + "/aisafety/output_slurm"
os.chdir(logPath)
print('Output of Slurm Jobs will be placed in:\t',logPath)

shPath = home + "/aisafety/new_march_21/code/"
print('Shell script is located at:\t',shPath)

if start == 0 and end == 66:
    mini = [2*i for i in range(32)] + ['64']
    maxi = [2*i +1 for i in range(32)] + ['66']
elif (end-start) > 2 and (end-start)%2 == 1:
    mini = [start + 2*i for i in range(int((end-start+1)/2))]
    maxi = [mini[i] + 1 for i in range(len(mini))]
else:
    mini = [start]
    maxi = [end]

print(mini)
print(maxi)

for k in range(len(mini)):
    
    submit_command = ("sbatch "
            "--job-name=AUCRank_{0}_{1}_{2}_{3}_{4} "
            "--export=NUM_F={0},MIN={1},MAX={2},MODE={3},PARAM={4},TRAINDATASET={5} {6}auc_ranking_attack_parallel.sh").format(NUM_DATASETS, mini[k], maxi[k], mode, param, traindataset, shPath)
    
    print(submit_command)
    exit_status = subprocess.call(submit_command, shell=True)
    if exit_status==1:  # Check to make sure the job submitted
        print("Job {0} failed to submit".format(submit_command))
print("Done submitting jobs!")