from InstaLoaderTommy import InstaloaderTommy
import redis
import time
from multiprocessing import Process

r = redis.Redis(decode_responses=True)

scraper = InstaloaderTommy()
# TODO: CHECK TIME SINCE SCRAPE
# TODO: ADD ALL BLOOM / CUCKOO FILTER

def scrape(tag):
    r.delete(f'scrape:finished:{tag}')
    with scraper:
        total = 0
        for count in scraper.get_hashtag_posts(
            tag, resume=True):
            total += count
            print(f'scraped: {total}')
            if r.xlen(f'tags:out:{tag}') > 5:
                r.lpush('cache:queue:ready', tag)
            if total > 600:
                total = 0
                res = r.lpush('cache:queue:ready', tag)
                print(f'push cache:queue:ready: {res}')
                break

# TODO: STOP SCRAPING WHEN DUPLICATE FOUND
# CUCKOO FILTER?
# store every prediction. if exists retrieve prediction else predict.
# TODO: VARY SCRAPING BY SIZE AND CONTENT OF HASHTAG FEED

while True:
    print('Ready! Waiting for a hashtag from redis...')
    tag = r.brpop('tagsin')[1]
    key = f'scraped:recent:{tag}'
    scraped = r.get(key)
    print(f'SCRAPE REQ: {tag}')
    # print(f'scraped_tag: {scraped}')
    if scraped:
        print(f'FOUND SCRAPED_RECENT KEY')
        continue
    start = time.perf_counter()
    p = Process(target=scrape, args=(tag,))
    p.start()
    end = time.perf_counter()
    print(f'TIMER: {end - start}')
    res = r.set(key, "True")
    r.expire(f'scraped:recent:{tag}', 600)

# TODO: check if tag has been recently scraped.
    
