
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=168
#SBATCH --mem-per-cpu=4591
#SBATCH --time=24:00:00

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 SciPy-bundle/2023.11

export p=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=1


# this script takes the source files and the script file, combines them and then runs it




python 1dAnderson_combined.py $p "$1"
