from redis import Redis
from time import sleep, perf_counter
from multiprocessing import Process
from math import ceil
from json import dumps
from redisbloom.client import Client
import time

r = Redis(decode_responses=True)
rb = Client()


def divide_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def cache(tag):
    print(tag)
    results = []
    stream_key = f'tags:out:{tag}'
    # added = r.sadd('set:cache:queue', tag)
    #         if added:
    #             r.lpush('cache:queue:ready', tag)

    # TODO: IF NO RESULTS IN 1ST TWO PAGES, RENDER A RELATED PAGE
    stream = r.xread({stream_key: b"0-0"}, block=10000)
    if stream:
        stream = stream[0][1]
    # TODO: BUILDING TWO LISTS...?
    # TODO: How many pages to cache
    for (_, post) in stream:
        results.append(
            {
                'id': post['postId'],
                'imgUrl': post['imgUrl'],
                'link': post['link'],
                'related': post['related_tag']
            }
        )
    per_page = 14
    master_list = list(divide_list(results, per_page))

    total_pages = ceil(len(stream) / per_page)
    total = len(stream)

    page = 1
    for p in master_list:
        json_result = dumps(p)
        # TODO: TEST: DIC v F-string
        page_response = f'{{"total":{total},"total_pages":{total_pages},"results":{json_result}}}'
        key = f'/enqueue?tag={tag}&page={page}'
        # TODO: Cache expire FORMULA. results count 5 - 5000
        key = r.set(key, page_response, ex=6600)
        page += 1
    # CACHE COMPLETE. DEL FROM QUEUE
    r.srem('set:cache:queue', tag)
    
# TODO WHATS UP
while True:
    print('Cache Machine Ready! Waiting for tags from redis...')
    # TODO: DONE: TOPK TOP100 REQUESTS
    min_results_to_cache = 5
    tag = r.brpop('cache:queue:ready')[1]
    print(f'cache request: {tag}')
    p = Process(target=cache, args=(tag,))
    p.start()
    # if r.xlen(f'tags:out:{tag}') > min_results_to_cache:
    #     p = Process(target=cache, args=(tag,))
    #     p.start()
    # else:
    #     r.lpush('cache:queue:ready', tag)
    #     continue
    # 
    print(f'Received cache request for {tag}')
    # rb.topkAdd('topk:10requests', tag)
    print(time.strftime('%X %x %Z'))
    
    
    
