#!/bin/bash

#SBATCH --partition alldlc2_gpu-l40s   # short: -p <partition_name>
#SBATCH --job-name DiT_w_token_cont          # short: -J <job name>

#SBATCH --output logs/%x-%A-concept_graphs.out   # STDOUT  %x and %A will be replaced by the job name and job id, respectively. short: -o logs/%x-%A-job_name.out
#SBATCH --error logs/%x-%A-concept_graphs.err    # STDERR  short: -e logs/%x-%A-job_name.out

#GET two nodes
# Define the amount of memory required per node
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --time=12:59:59


cd /work/dlclarge2/aliy-maskgit/concept_graphs

echo "Workingdir: $PWD";
echo "Started at $(date)";

# A few SLURM variables
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Activate your environment
# You can also comment out this line, and activate your environment in the login node before submitting the job
. ~/.bashrc # Adjust to your path of Miniconda installation
conda activate vqgan

#move dataset to tmp using bash file in the current directory called dataset_to_tmp.sh
# Running the job
start=`date +%s`

python train_merge.py --run_desc XAI  --pixel_size 28 --our_labels False  --model DiT  --dropout 0.0 --batch_size 256 --n_epoch 6000 --token_folder /work/dlclarge2/aliy-maskgit/maskgit/image_tokenization/vqgan_logs/2025-02-13T13-25-14_codebook_Third_synthetic_DLC13913381
end=`date +%s`
runtime=$((end-start))


echo Job execution complete
echo Runtime: $runtime