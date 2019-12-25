import redis

conn = redis.Redis(host='localhost', port=6379, db=0)
if not conn.ping():
        raise Exception('Redis unavailable')

# Load the RedisAI model
print('Loading model - ', end='')
with open('/home/tommy/development/projects/scrape/unbalanced4.6.onnx', 'rb') as f:
    model = f.read()
    res = conn.execute_command(
        'AI.MODELSET', 'auction:model', 'ONNX', 'CPU', model)
    print(res)

initialized_key = '{}:initialized'.format('post:')
# Check if this Redis instance had already been initialized
initialized = conn.exists(initialized_key)
if initialized:
    print('Discovered evidence of a privious initialization - skipping.')
    exit(0)
# Load the gear    
print('Loading gear - ', end='')
with open('gear.py', 'rb') as f:
    gear = f.read()
    res = conn.execute_command('RG.PYEXECUTE', gear)
    print(res)
print('Flag initialization as done - ', end='') 
print(conn.set(initialized_key, 'most certainly.'))