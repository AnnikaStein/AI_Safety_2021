#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1

#SBATCH --ntasks-per-node=1

#SBATCH --mem-per-cpu=10G

#SBATCH --cpus-per-task=1

#SBATCH --job-name=defaults49

#SBATCH --output=output.%J.txt

#SBATCH --time=0:30:00

#SBATCH --account=rwth0583

# with gpu: 

# SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de

cd /home/um106329/aisafety/april_21
source ~/miniconda3/bin/activate
conda activate my-env
python3 files_defaults.py 49 '0.001'
