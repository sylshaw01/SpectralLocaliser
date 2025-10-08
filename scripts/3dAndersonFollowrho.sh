#!/bin/bash


cat > job3dAadaptiverho.sh << "EOF"
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=168
#SBATCH --mem-per-cpu=4591
#SBATCH --time=48:00:00

module purge
module load GCC/13.2.0 OpenMPI/4.1.6 SciPy-bundle/2023.11

export p=$SLURM_CPUS_PER_TASK
export SLURM_CPU_BIND=none
export OMP_NUM_THREADS=1






python 3dAndersonadaptive_rho_combined.py $p "$1"
EOF

python - <<"EOF"
import re
import sys

# read SLModels
with open("../src/SLmodels.py", "r") as f:
    slmodels_code = f.read()

# read 1dAnderson
with open("3dAndersonFollowrho.py", "r") as f:
    anderson_code = f.read()

anderson_code = re.sub(r"^sys\.path\.append\(['\"]\.\.\/src['\"]\)$", "", anderson_code, flags=re.MULTILINE)

anderson_code = re.sub(r"^from SLmodels import \*$", "", anderson_code, flags=re.MULTILINE)


combined = f"""#!/usr/bin/env python3

{slmodels_code}

{anderson_code}
"""

with open("3dAndersonadaptive_rho_combined.py", "w") as f:
    f.write(combined)
EOF

sbatch job3dAadaptiverho.sh $1




