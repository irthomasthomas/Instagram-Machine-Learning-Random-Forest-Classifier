import redis
from sys import argv
from rejson import Client as rjclient
from rejson import Path

rj = rjclient(decode_responses=True)
rdb = redis.Redis(decode_responses=True)
input = argv[1]
startkey = f'json:page:{input}'

page = rj.jsonget