redis-cli flushall
killall redis-server
gnome-terminal -x sh -c "redis-server ~/dev/redis/redis.conf"
python3 init.py
