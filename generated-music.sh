#!/bin/bash

cd /home/icowen/GeneratedMusic

sbatch \
--nodes=1 \
--ntasks-per-node=20 \
--mem=32000 \
--time=01:00:00 \
--job-name=generatedMusic \
--output=output.log

python3 TwoLetterNeuralNet.py
