cat ~/src/reinforcement_exp/output/output.log  | grep "output:" | sed 's/output: //g' | sed 's/ //g' > ~/src/reinforcement_exp/output/output.csv
source activate reinf && python ~/src/reinforcement_exp/visualization/visualize.py ~/src/reinforcement_exp/output
