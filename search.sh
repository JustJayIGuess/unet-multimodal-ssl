#!/bin/zsh

# Magic number 1337 for seed to prove no cherry-picking.
# Will hopefully do this for multiple seeds anyway though.

for sample in {0..20}; do
    for lam in {0..1}; do
        ~/Documents/Code/Tensorflow/env/bin/python run-search.py $lam 1337
    done
done