#!/bin/bash
#SBATCH --job-name=findr-array
#SBATCH --output=/scratch/gpfs/tdkim/findr/npx_luo/logs/slurm-%A.%a.out
#SBATCH --error=/scratch/gpfs/tdkim/findr/npx_luo/logs/slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:29:00
#SBATCH --array=1-225
#SBATCH --mail-type=all
#SBATCH --mail-user=tdkim@princeton.edu

datafolderpath="/scratch/gpfs/tdkim/findr/npx_luo/manuscript2023a/recordingsessions/2024_04_09"
analysisfolderpath="/scratch/gpfs/tdkim/findr/npx_luo/manuscript2023a/fits/2024_04_09"

shopt -s nullglob
session_id=0
file_array=(${datafolderpath}/*.npz)
sess_names=("${file_array[@]##*/}")
sess_names=("${sess_names[@]%.*}")
datafilepath="${file_array[${session_id}]}"
analysispath="${analysisfolderpath}/${sess_names[${session_id}]}/config_findr_${SLURM_ARRAY_TASK_ID}"

module purge
module load anaconda3/2022.5 cudatoolkit/11.7 cudnn/cuda-11.x/8.2.0
source activate findr
python main.py --datapath=$datafilepath --workdir=$analysispath --config=configs/default.py:$SLURM_ARRAY_TASK_ID
