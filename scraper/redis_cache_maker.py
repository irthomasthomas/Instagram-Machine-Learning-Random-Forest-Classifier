from redis import Redis
from time import sleep, perf_counter
from multiprocessing import Process
from math import ceil
from json import dumps
from redisbloom.client import Client

r = Redis(decode_responses=True)
rb = Client()


def divide_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cache(tag):
    results_list = []
    stream = f'tags:out:{tag}'
    result = r.xread({stream:b"0-0"}, block=10000)
    # print(f'tag:{tag}')
    print(f'stream:{stream}')

    if len(result) <= 28:
        sleep(1)
        result = r.xread({stream:b"0-0"})
        # print(f'len:{len(result)}')
        # print('RESULT LESS THAN 28')
        # print(f'RESULT: {result}')
    if result:
        print(f'RESULT OK')
        result = result[0][1]
    for (_,rN) in result:
        results_list.append(rN)
    
    per_page = 14
    master_list = list(divide_list(results_list, per_page))
    total_pages = ceil(len(result) / per_page)
    total = len(result)

    page = 1
    for p in master_list:
        # json_result = dumps(p)
        if page == 1:
            page += 1
            continue
        json_result = dumps(p)
        message = f'{{"total":{total},"total_pages":{total_pages},"results":{json_result}}}'
        key = f'/enqueue?tag={tag}&page={page}'
        # print(f'key: {key}')
        # print(f'message: {message}')   
        key = r.set(key,message)
        page += 1
        # print(f'key: {key}')     
    

while True:
    print('Cache Machine Ready! Waiting for tags from redis...')
    # tag = r.brpop('cache:queue')
    # TODO: TOPK REQUESTS
    tag = r.brpop('cache:queue:ready')[1]
    rb.topkAdd('top10requests', tag)
    key = f'scraped:recent:{tag}'
    scraped_recently = r.get(key)
    if scraped_recently:
        print(f'found recent tag skipping cache ')
        continue
    print(f'Received cache request for {tag}')
    start = perf_counter()
    p = Process(target=cache, args=(tag,))
    p.start()
    end = perf_counter()
    # print(f'TIMER: {end - start}')


# TODO: check if tag has been recently scraped.
    
