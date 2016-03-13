cat ~/src/reinforcement_learning/output/output.log  | grep "output:" | sed 's/output: //g' | sed 's/ //g' > ~/src/reinforcement_learning/output/output.csv
source activate reinf && python ~/src/reinforcement_learning/visualization/visualize.py ~/src/reinforcement_learning/output
