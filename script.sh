#!/bin/bash
echo "-----Computing Baselines-----"
python compute_overlap_baseline.py SICK_trial.txt>results.txt
R --no-save --slave --vanilla --args results.txt SICK_trial.txt <sick_evaluation.R
echo "-----Baselines end------"

echo "---My code---"
echo "Training"
python train.py
echo "Testing"
rm -f results.txt
python test.py
echo "Results: "
R --no-save --slave --vanilla --args results.txt SICK_trial.txt <sick_evaluation.R
