# TODO: Hashtag sets
def addToLikesZset(x):
    # execute('SADD', 'likes', x['key'])
    execute('ZADD', 'likes', x['likes'], x['key'])
    return x

GB().filter(lambda x: x['key'] != 'likes').foreach(
    addToLikesZset).register('*')
    