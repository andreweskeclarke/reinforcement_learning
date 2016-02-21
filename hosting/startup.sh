cp index.html /www/data/
sudo kill `cat ./tmp/pids/unicorn.pid`
sudo /etc/init.d/nginx stop
sudo kill `cat ./tmp/pids/nginx.pid`
sudo cp nginx.conf /etc/nginx/nginx.conf
mkdir -p tmp
mkdir -p tmp/pids
mkdir -p tmp/sockets
mkdir -p log
sudo /etc/init.d/nginx start
