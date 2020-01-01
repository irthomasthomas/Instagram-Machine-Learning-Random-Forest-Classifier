# import redis
from time import sleep
from json import dumps
from math import ceil

# TODO: add a key when there are 10+ results 
# TODO: GEAR TO GENERATE CACHE PAGES WITH EXPIRING KEY
# request page /enqueue/tag=bubblecap&page=1

def divide_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cache_pages(x):
    print('VER six')
    results = []
    tag = execute('RPOP', 'cache:queue')
    stream = f'tags:out:{tag}'
    print(f'stream: {stream}')
    
    # total = 0
    # i = 1

    # while total < 30:
    #     total = execute('XLEN', f'tags:out:{tag}')
    #     print(f'total: {total}')
    #     i += 1
    #     if (total > 28) or (i > 10):
    #         break
    #     else:
    #         sleep(1)
    
    total = execute('XLEN', f'tags:out:{tag}')
    print("BREAK")
    print(f'total: {total}')
    result = execute('XREAD', {stream:b"0-0"})[0][1]
    print(f'result: {result}')

    for (_,r) in result:
        results.append(r)
    pages = 14
    L = list(divide_list(results, pages))
    total_pages = ceil(total / pages)

    page = 1
    for i in L:
        json_result = dumps(i)
        message = f'{{"total":{total},"total_pages":{total_pages},"results":{json_result}}}'
        key = f'/enqueue?tag={tag}&page={page}'
        print(f'key: {key}')
        print(f'message: {message}')
        execute('set', key, message) 
        page += 1
    
gb = GearsBuilder()
gb.filter(lambda    x: x['key'] == 'cache:queue')		
gb.map(cache_pages)
gb.register()
