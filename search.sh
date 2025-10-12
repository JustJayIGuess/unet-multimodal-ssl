#!/bin/zsh

for num_lab in {1,3,5,10,15,30,60,90}; do
    for fold in {0..4}; do
        for seed in {0}; do
            for lam in {0..1}; do
                ~/Documents/Code/Tensorflow/env/bin/python model.py $lam $seed $fold "$num_lab labelled " $num_lab
            done
        done
    done
done