ls -t /www/data/ | grep weight | tail -n +6 | xargs -I {} rm /www/data/{}
ls -t /www/data/ | grep model | tail -n +6 | xargs -I {} rm /www/data/{}
