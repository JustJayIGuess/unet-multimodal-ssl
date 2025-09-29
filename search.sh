#!/bin/zsh

# for seed in {0..3}; do
for sample in {0..6}; do
    for lam in {0..3}; do
                # echo "running with lam[$lam], seed=1337"
        ~/Documents/Code/Tensorflow/env/bin/python run-search.py $lam 1337
    done
done
# done