scp andrew@aeclarke.com:/www/data/* output/ && cat output/output.csv | tail -n +2 | sed 's/\(.*\)/Average: \1/g' > output/output.log
