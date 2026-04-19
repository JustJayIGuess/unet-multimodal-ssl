rm cache/*
mpirun -n 4 ~/Documents/Code/Tensorflow/env/bin/python model.py 32 -l 32 -n "l32b w1 i0" -s -1 -p 30 -w 1.0 -f 4 --cache-disk
