#!/bin/bash 
#SBATCH -J seeded
## SBATCH -p edr,fdr
#SBATCH -t 24:00:00
## SBATCH --mem-per-cpu 4000
#SBATCH -N 1 -n 32 --mem=0 -p edr
## SBATCH -N 1
## SBATCH -n 8
## SBATCH --exclusive
#SBATCH -e error.%x
#SBATCH -o stdout.%x
#SBATCH --no-requeue
#############
## case info

echo run begins at: `date`
echo Directory: `pwd`

## export casename=_CASENAME_
## export currentiter=_CURRENTITER_
## export eachiter=_EACHITER_
## export totaliter=_TOTALITER_
## export nnodes=_NNODES_
## export ncores=_NCPUS_

echo 'Your job is running on node(s):'
echo $SLURM_JOB_NODELIST
echo 'Cores per node:'
echo $SLURM_TASKS_PER_NODE

## environment:
if test -f /etc/profile.d/modules.sh    ; then . /etc/profile.d/modules.sh    ; fi
if test -f /etc/profile.d/zz_modules.sh ; then . /etc/profile.d/zz_modules.sh ; fi
if [[ ${SLURM_JOB_PARTITION} == *"hdr"* ]]; then
  module purge
  module load intel/2021.4.0_rhel8
  module load openmpi/3.1.6
elif [[ ${SLURM_JOB_PARTITION} == *"fdr"* || ${SLURM_JOB_PARTITION} == *"edr"* ]]; then
  module add anaconda3/2020.11
fi
echo "list of loaded modules:"
module list
echo " "

source activate dedalus-test

python3 m2d_WC1_seeded.py
