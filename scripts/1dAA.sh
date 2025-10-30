#!/bin/bash


cat > jobAA.sh << "EOF"
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=84
#SBATCH --mem-per-cpu=4591
#SBATCH --time=48:00:00

module purge
module load OpenMPI/5.0.8-GCC-14.3.0

export p=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=1


export INSTALL_DIR="$HOME/slepc-gnu-install"
export PETSC_DIR="$INSTALL_DIR/petsc-main"
export PETSC_ARCH="arch-linux-c-opt"
export SLEPC_DIR="$INSTALL_DIR/slepc-main"


source "$INSTALL_DIR/slepc_env/bin/activate"






python 1dAA_combined.py $p "$1"
EOF

python - <<"EOF"
import re
import sys

# read SLModels
with open("../src/SLmodels.py", "r") as f:
    slmodels_code = f.read()

# read 1dAA
with open("1dAA.py", "r") as f:
    AA_code = f.read()

AA_code = re.sub(r"^sys\.path\.append\(['\"]\.\.\/src['\"]\)$", "", AA_code, flags=re.MULTILINE)

AA_code = re.sub(r"^from SLmodels import \*$", "", AA_code, flags=re.MULTILINE)


combined = f"""#!/usr/bin/env python3

{slmodels_code}

{AA_code}
"""

with open("1dAA_combined.py", "w") as f:
    f.write(combined)
EOF

sbatch jobAA.sh $1




