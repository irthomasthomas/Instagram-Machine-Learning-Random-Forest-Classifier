import redis
import string

conn = redis.Redis(host="localhost", port=6379, db=0)

def storeResults(x):
    ''' store to output stream '''
    execute('SADD', 'allposts', x['key'])
    return x

gb = GearsBuilder()
gb.map(lambda x: print(x))

gb.register('tagsin')
# gb.map(storResults)


# https://www.instagram.com/p/Bt3QrphggBw/media/?size=m
