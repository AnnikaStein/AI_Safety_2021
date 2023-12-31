#!/usr/local_rwth/bin/zsh

#SBATCH --ntasks=1

#SBATCH --ntasks-per-node=1

# SBATCH --mem-per-cpu=160G

#SBATCH --cpus-per-task=1

# SBATCH --job-name=TTtr49_0_30_new_0.001

#SBATCH --output=output.%J.txt

# SBATCH --time=65:30:00

#SBATCH --account=rwth0583

# with gpu: 

# SBATCH --gres=gpu:1

#SBATCH --mail-type=ALL

#SBATCH --mail-user=annika.stein@rwth-aachen.de


cd /home/um106329/aisafety/june_21/evaluate
module unload intelmpi; module switch intel gcc
module load cuda/11.0
module load cudnn
source ~/miniconda3/bin/activate
conda activate my-env
# echo $START $END $COMPARESETUP $WM
# python eval_roc_new.py '278' 31,40 BvL_eps0.01 'no' ${WM} '0.001' '-1' 'yes' 'yes' '-1' 'yes' 'AUC'
# python eval_roc_new.py '278' \'${START},${END}\' ${COMPARESETUP} 'no' ${WM} '0.001' '-1' 'yes' 'yes' '-1' 'yes' 'AUC'
python eval_roc_new.py ${FILES} ${START},${END} ${COMPARESETUP} ${PLOTDEEPCSV} ${WM} ${DEFAULT} ${JETS} ${DOMINIMAL_EVAL} ${ADD_AXIS} ${FGSM_SETUP} ${LOG_AXIS} ${SAVE_MODE}
