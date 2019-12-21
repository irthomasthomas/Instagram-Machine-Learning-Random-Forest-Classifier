def addToUpdatedZset(x):
    import time
    now = time.time()
    execute('ZADD', 'updated', now, x['key'], '1')
    return x


GB().filter(lambda x: x['key'] != 'updated').foreach(
    addToUpdatedZset).register('*')

# def extract_hashtags(caption):
#     tag_regex = re.compile(
#         r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
#     tags = re.findall(hashtag_regex, caption.lower())
#     execute('SADD', 'posttags'+str(id),*tags)    
