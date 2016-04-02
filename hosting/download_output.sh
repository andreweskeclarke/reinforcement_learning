scp -i ~/deep_learning.pem ubuntu@52.53.253.15:/home/ubuntu/src/reinforcement_learning/output/output.csv /www/data/
scp -i ~/deep_learning.pem ubuntu@52.53.253.15:/home/ubuntu/src/reinforcement_learning/output/conv* /www/data/
scp -i ~/deep_learning.pem ubuntu@52.53.253.15:/home/ubuntu/src/2reinforcement_learning/output/output.csv /www/data/secondary/
scp -i ~/deep_learning.pem ubuntu@52.53.253.15:/home/ubuntu/src/2reinforcement_learning/output/conv* /www/data/secondary/
awk 'NR == 1 || NR % 10 == 0' /www/data/output.csv > /www/data/output_small.csv
