#!/bin/bash
#SBATCH --job-name=Prop.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=sxmq
#SBATCH --nodelist=sxm[002]
#SBATCH --output=./SLURM/R-%SLURM.%j.out

source /home/parsas1/conda/bin/activate
conda activate pt_dcam
python -u independent_runs.py --cancer_type=$cancer_type --exp_num=$exp_num --gpu_num=$gpu_num