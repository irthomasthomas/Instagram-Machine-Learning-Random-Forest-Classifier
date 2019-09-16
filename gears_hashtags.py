# r.sadd("posttags:"+id,tag)
#     r.sadd("tagposts:"+tag,id)


def myfilter(k):
    # import sys
    try:
        return not (k.startswith('all_keys'))
    except Exception as e:
        with open("/root/redisgears.log", "w") as f:
            f.write(str(e))
            # f.write("Unexpected error: " + sys.exc_info()[0])
    

def addHastagSet(x):
    # return x
    # if "all_keys" in x['key']:
    #     return x
    if "posttags" in x['key']:
        return
    import re
    hashtag_regex = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
    caption = execute('HGET', x['key'], 'caption')
    caption = caption.lower()
    tags = re.findall(hashtag_regex, caption)
    # execute('SADD', 'all_captions', caption)
    execute('sadd', 'posttags:'+x['key'], *tags)
    
    # execute('sadd', 'all_keys', x['key'])

    # return x
    # # for tag in tags:
    # execute('sadd', 'posttags:'+x['key'], *tags)
    # return x

# def test(x):
#     execute('sadd', 'all_keys', x['key'])
#     return x

# builder = GearsBuilder()
# builder.filter(lambda x: x['key'] != 'all_keys')
# # builder.map(lambda x: test)
# builder.foreach(test)
# builder.register()
builder = GearsBuilder()
builder.filter(lambda x: x['key'] != 'all_keys')
# builder.filter(myfilter)
builder.foreach(addHastagSet) # does not emit anything (so no return needed?)
# builder.map(lambda x: execute('sadd', 'all_keys', x['key']))
builder.register()

# GearsBuilder().foreach(lambda x: redisgears.execute_command(
#     'set', x['value'], x['key'])) 
#     # will save value as key and key as value