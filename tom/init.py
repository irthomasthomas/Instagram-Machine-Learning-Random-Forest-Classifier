from redis import Redis
from redisbloom.client import Client

rb = Client()
rdb = Redis(host='localhost', port=6379, db=0)
if not rdb.ping():
        raise Exception('Redis unavailable')

# Load the RedisAI model
print('Loading model - ', end='')
with open('/home/tommy/development/projects/scrape/unbalanced4.6.onnx', 'rb') as f:
    model = f.read()
    res = rdb.execute_command(
        'AI.MODELSET', 'auction:model', 'ONNX', 'CPU', model)
    print(res)

initialized_key = '{}:initialized'.format('post:')
# Check if this Redis instance had already been initialized
initialized = rdb.exists(initialized_key)
if initialized:
    print('Discovered evidence of a privious initialization - skipping.')
    exit(0)
# Load the gear    
print('Loading gear - ', end='')
with open('gear.py', 'rb') as f:
    gear = f.read()
    res = rdb.execute_command('RG.PYEXECUTE', gear)
    print(res)

# rb.topkAdd('top10tags', tag)
if not rdb.exists('topk:10tags'):
    rb.topkReserve('topk:10tags', 10, 100, 5, 0.9)
# rb.topkAdd('top100tags', tag)
if not rdb.exists('topk:100tags'):
    rb.topkReserve('topk:100tags', 100, 2000, 6, 0.9)
# rb.topkAdd('top10requests', tag)
if not rdb.exists('topk:10requests'):
    rb.topkReserve('topk:10requests', 10, 100, 5, 0.9)
if not rdb.exists('topk:100requests'):
    rb.topkReserve('topk:100requests', 100, 2000, 6, 0.9)            
# execute('zincrby', 'topzhashtags:', rootTag, 1) # created automatic

# sketch:out not being used at present
# execute('cmsIncrBy', f'sketch:out:{rootTag}', postId, '1')    

# execute('cmsIncrBy', 'sketch:allposts', postId, '1')
if not rdb.exists('sketch:allposts'):
    rb.cmsInitByDim('sketch:allposts', 2000, 10)
if not rdb.exists('sketch:predicted'):
    rb.cmsInitByDim('sketch:predicted', 2000, 10)
# TODO: CMS complexity is the product...
# of w,d and the width of the counters
# e.g. a sketch with 0.01% error rate
# probability 0.01% is created using 10
# hash functions and 2000-counter arrays.
# 16bit counters the overall memory requirement
# is 40KB. Reading and writing in O(1) constant time


# used in scraper as tag_sketch_key
# execute('cmsIncrBy', f'sketch:all:{rootTag}', postId, '1')


print('Flag initialization as done - ', end='') 
print(rdb.set(initialized_key, 'most certainly.'))