#!/bin/bash

mkdir -p "experiments"

args="-learningRate 0.01 -batchSize 100 -weightDecay 0.00005 -momentum 0.9 -classifierFile \"experiments/classifier_1.table\" -resultsFile \"experiments/results_1.table\""

if [ "$1" = "continue" ]; then
./run.lua -load -maxSeconds $2 $args 
else
./run.lua -maxSeconds $2 $args 
fi
