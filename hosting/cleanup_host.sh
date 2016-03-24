ls -t /home/ubuntu/src/reinforcement_learning/output | grep "weight.*\.h5" | tail -n +6 | xargs -I {} rm /home/ubuntu/src/reinforcement_learning/output/{}
ls -t /home/ubuntu/src/reinforcement_learning/output | grep "model.*\.json" | tail -n +6 | xargs -I {} rm /home/ubuntu/src/reinforcement_learning/output/{}
