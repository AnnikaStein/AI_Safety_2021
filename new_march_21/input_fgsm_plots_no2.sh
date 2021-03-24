#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=2

#SBATCH --ntasks-per-node=2

#SBATCH --mem-per-cpu=16G

#SBATCH --cpus-per-task=2

#SBATCH --job-name=CP_NOISE_0

#SBATCH --output=output.%J.txt

#SBATCH --time=20:30:00

#SBATCH --account=rwth0583

# with gpu: 

# SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de

cd /home/um106329/aisafety
#module unload intelmpi; module switch intel gcc
#module load cuda/110
#module load cudnn
source ~/miniconda3/bin/activate
conda activate my-env
python3 compare_weighted.py 12 21 'fgsm' 'yes'
