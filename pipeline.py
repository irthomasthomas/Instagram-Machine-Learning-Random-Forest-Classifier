import redis
from rq import Queue


r = redis.Redis(host='localhost', port=6379, db=0)
q = Queue(connection=r)
redis_key = "tagqueue"

def queuejob(hashtag):
    q = Queue("scrapetag", connection=r)
    job = q.enqueue(processimg, src_path)


# scrape hashtags to redis 1
# push to stream

# Clean text
# Model
# fetch img
# fetch comments
# Extract price
# Publish
