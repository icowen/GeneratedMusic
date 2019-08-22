#!/bin/bash

cd /home/icowen/GeneratedMusic

sbatch \
--nodes=1 \
--ntasks-per-node=20 \
--mem=32000 \
--time=01:00:00 \
--job-name=generatedMusic \
--output=output.log

pip3 install tensorflow-gpu --user
pip3 install keras --user
python3 TwoLetterNeuralNet.py
