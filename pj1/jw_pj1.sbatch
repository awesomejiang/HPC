#SBATCH --job-name=jwjiang
#SBATCH --output=jwjiang.output
#SBATCH --cpus-per-task=8
#SBATCH --partition=sandyb
module load g++6.1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
./