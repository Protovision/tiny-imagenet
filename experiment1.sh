#!/bin/bash

mkdir -p "experiments"

args="-maxEpochs 200 -learningRate 0.01 -batchSize 256 -testBatchSize 1000 -weightDecay 0.0005 -momentum 0.9 -learningRateCorrectionFactor 10 -classifierFile \"experiments/classifier_2.table\" -resultsFile \"experiments/results_2.table\""

if [ "$1" = "continue" ]; then
./run.lua -load -maxSeconds $2 $args 
else
./run.lua -maxSeconds $2 $args 
fi
