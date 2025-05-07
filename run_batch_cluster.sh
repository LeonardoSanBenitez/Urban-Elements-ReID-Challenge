#!/bin/sh
#SBATCH --job-name=reid                             # Name of the job
#SBATCH --partition=gpu                             # Use the GPU partition; Existing partitions: gpu, cpu, gpu_long
#SBATCH --gres=gpu:a100:1	                        # Select machine config (gpu:v100:1 = 1 V100 16GB GPU, gpu:a100:1 = 1 A100 40GB GPU)
#SBATCH --time=12:00:00                             # Set maximum run time for the job (hh:mm:ss)
#SBATCH --output=assets/cluster_jobs/%j-train-out   # Redirect stdout
#SBATCH --error=assets/cluster_jobs/%j-train-err    # Redirect stderr

if [ "$(basename "$PWD")" != "Urban-Elements-ReID-Challenge" ]; then
    echo "Error: This script must be run from the 'Urban-Elements-ReID-Challenge' directory."
    exit 1
fi

cd "Code/Files for PAT/"
srun apptainer exec --nv ../../run_cluster.sif python run_all.py
