# TODO: Hashtag sets
def addToUpdatedZset(x):
    import time
    now = time.time()
    execute('ZADD', 'updated', now, x['key'], '1')
    return x


GB().filter(lambda x: x['key'] != 'updated').foreach(
    addToUpdatedZset).register('*')
