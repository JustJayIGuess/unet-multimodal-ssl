#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=384GB
#PBS -l jobfs=256GB
#PBS -q gpuvolta
#PBS -P w09
#PBS -l walltime=00:05:00
#PBS -l storage=gdata/w09+scratch/w09
#PBS -m abe
#PBS -M u7922560@anu.edu.au
#PBS -l wd

module load python3/3.12.1
module load openmpi/5.0.8

source ../../tf20env/bin/activate

rm cache/*
mpirun -n 4 ~/Documents/Code/Tensorflow/env/bin/python model.py 32 -l 32 -n "l32b w1 i0" -s 0 -p 30 -w -1 -c 4
