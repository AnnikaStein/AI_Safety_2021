#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1

#SBATCH --ntasks-per-node=1

#SBATCH --mem-per-cpu=187200M

#SBATCH --cpus-per-task=1

# SBATCH --job-name=AUCranking

#SBATCH --output=output.%J.txt

# SBATCH --time=4:00:00

#SBATCH --account=rwth0583

# with gpu: 

# SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de

cd /home/um106329/aisafety/new_march_21/code
source ~/miniconda3/bin/activate
conda activate my-env
python3 auc_outputs_tt_attack.py ${NUM_F} ${MODE} ${PARAM}