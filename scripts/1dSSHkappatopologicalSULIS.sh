#!/bin/bash

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 SciPy-bundle/2023.11


cat > job1dSSH.sh << "EOF"
#!/bin/bash


#SBATCH --account=su007-rr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=84
#SBATCH --mem-per-cpu=3850
#SBATCH --time=48:00:00


module purge
module load GCC/13.2.0 OpenMPI/4.1.6 SciPy-bundle/2023.11

export p=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=1


# export INSTALL_DIR="$HOME/slepc-gnu-install"
# export PETSC_DIR="$INSTALL_DIR/petsc-main"
# export PETSC_ARCH="arch-linux-c-opt"
# export SLEPC_DIR="$INSTALL_DIR/slepc-main"


# source "$INSTALL_DIR/slepc_env/bin/activate"






python 1dSSH_combined.py $p "$1"
EOF

python - <<"EOF"
import re
import sys

# read SLModels
with open("../src/SLmodels.py", "r") as f:
    slmodels_code = f.read()

# read 1dSSH
with open("1dSSH-kappa-test-topological-transition.py", "r") as f:
    SSH_code = f.read()

SSH_code = re.sub(r"^sys\.path\.append\(['\"]\.\.\/src['\"]\)$", "", SSH_code, flags=re.MULTILINE)

SSH_code = re.sub(r"^from SLmodels import \*$", "", SSH_code, flags=re.MULTILINE)


combined = f"""#!/usr/bin/env python3

{slmodels_code}

{SSH_code}
"""

with open("1dSSH_combined.py", "w") as f:
    f.write(combined)
EOF

sbatch jobSSH.sh $1




