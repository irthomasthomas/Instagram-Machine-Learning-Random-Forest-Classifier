import sys
import redis
from redisbloom.client import Client

if len(sys.argv) > 1:
     item_to_del=sys.argv[1]
     print(item_to_del)
else:
    print("You need to provide the item key to delete")
    exit(0)

r = redis.Redis(decode_responses=True)
print(f'redis: {str(r)}')
rb = Client()

# streamKey = f'tags:out:{rootTag}'
# del streamkey

# rb.topkReserve('top10tags2', 10, 10, 5, 0.9)

for key in r.scan_iter('tags:out:*'):
    tag = key.split(':')[2]
    len = r.xlen(key)
    print(f'{tag} {len}')
    # execute('topk.add', 'top100tags', rootTag)
    # TOPK.INCRBY key item increment 
    res = r.execute_command('topk.incrby', 'top10tags2', tag, len)
    print(res)

