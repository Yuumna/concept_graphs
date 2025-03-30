#!/bin/bash

#SBATCH --partition lmbhiwidlc_gpu-rtx2080   # short: -p <partition_name>
#SBATCH --job-name UNet_w/o_tok_cont           # short: -J <job name>

#SBATCH --output logs/%x-%A-concept_graphs_ol.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-concept_graphs_ol.err    # STDERR  short: -e logs/%x-%A-job_name.out

#GET node
# Define the amount of memory required per node
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:59:59


cd /work/dlclarge2/aliy-maskgit/concept_graphs

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
. ~/.bashrc # Adjust to your path of Miniconda installation
conda activate newmask

#move dataset to tmp using bash file in the current directory called dataset_to_tmp.sh
# Running the job
start=`date +%s`
python train_merge.py --run_desc first_run --our_labels False  --model U-Net  --batch_size 256 --n_epoch 6000
end=`date +%s`
runtime=$((end-start))


echo Job execution complete
echo Runtime: $runtime