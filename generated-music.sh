#!/bin/bash

sbatch \
--nodes=1 \
--ntasks-per-node=10 \
--mem=16000 \
--time=15:00 \
--job-name=generatedMusic \
--output=output.log

cd /home/icowen/GeneratedMusic

python KerasTest.py
