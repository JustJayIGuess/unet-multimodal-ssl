#!/bin/zsh

# Magic number 1337 for seed to prove no cherry-picking.
# Will hopefully do this for multiple seeds anyway though.

for sample in {0..5}; do
    for lam in {0..1}; do
        for sample in {0..2} do
            ~/Documents/Code/Tensorflow/env/bin/python model.py $lam 1337 $sample "15l-15ep-crossval "
        done
    done
done