import redis

conn = redis.Redis(host='localhost', port=6379, db=0)
if not conn.ping():
        raise Exception('Redis unavailable')

# Load the RedisAI model
print('Loading model - ', end='')
with open('/root/dev/projects/scrape/unbalanced4.6.onnx', 'rb') as f:
    model = f.read()
    res = conn.execute_command('AI.MODELSET', 'auction:model', 'ONNX', 'CPU', model)
    print(res)

# Load the gear    
print('Loading gear - ', end='')
with open('gear.py', 'rb') as f:
    gear = f.read()
    res = conn.execute_command('RG.PYEXECUTE', gear)
    print(res)
