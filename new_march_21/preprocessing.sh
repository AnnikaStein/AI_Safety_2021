#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=2

#SBATCH --ntasks-per-node=2

#SBATCH --mem-per-cpu=24G

#SBATCH --cpus-per-task=2

#SBATCH --job-name=PREP

#SBATCH --output=output.%J.txt

#SBATCH --time=10:10:00

#SBATCH --account=rwth0583

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de

cd /home/um106329/aisafety/new_march_21/code/preprocessing

source ~/miniconda3/bin/activate
conda activate my-env
python3 preprocessing.py

