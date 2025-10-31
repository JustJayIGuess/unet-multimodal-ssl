#!/bin/zsh

for num_lab in {60..60}; do
    for fold in {0..4}; do
#        for seed in {0}; do
        for lam in {0..1}; do
            ~/Documents/Code/Tensorflow/env/bin/python model.py $lam 0 $fold "$num_lab labelled " $num_lab
        done
#        done
    done
done
