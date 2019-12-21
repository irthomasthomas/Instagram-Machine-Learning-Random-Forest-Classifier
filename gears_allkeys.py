builder = GearsBuilder()
builder.filter(lambda x: x['key'] != 'all_keys')
builder.map(lambda x: execute('sadd', 'all_keys', x['key']))
builder.register()