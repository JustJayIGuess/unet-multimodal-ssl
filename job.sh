#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=384GB
#PBS -l jobfs=256GB
#PBS -q gpuvolta
#PBS -P w09
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/w09+scratch/w09
#PBS -m abe
#PBS -M u7922560@anu.edu.au
#PBS -l wd

module load cuda/12.5.1
module load cudnn/9.5.0-cuda12
module load python3/3.12.1
module load openmpi/5.0.8

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

source ../../tf20env/bin/activate

rm cache/*
mpirun -n 4 python model.py 32 -l 32 -n "l32b w1 i0" -s 0 -p 20 -w -1 -c 4
zip -r logs.zip logs
rm -r logs
