#!/bin/bash


cat > jobSSH.sh << "EOF"
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




python 1dSSH_combined.py $p "$1"
EOF

python - <<"EOF"
import re
import sys

# read SLModels
with open("../src/SLmodels.py", "r") as f:
    slmodels_code = f.read()

# read 1dSSH
with open("1dSSH.py", "r") as f:
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




