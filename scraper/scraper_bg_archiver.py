from InstaLoaderTommy import InstaloaderTommy
import redis
import time
from multiprocessing import Process

r = redis.Redis(decode_responses=True)

scraper = InstaloaderTommy()

def archive_scrape(tag):
    print(f'archive scraper {tag}')
    num_to_scrape = 600
    with scraper:
        total = 0
        try:
            for count in scraper.get_hashtag_posts(
                    tag, dump_page=True, resume=True,
                    duplicate_check=False, archive_run=True):
                total += count
                print(f'scraped total: {total}')
                added = r.sadd('set:cache:queue', tag)
                if added:
                    r.lpush('cache:queue:ready', tag)
                if total > num_to_scrape:
                    total = 0
                    exists = r.sadd('set:tags:archive', tag)
                    if exists:
                        r.lpush('archive:tagsin', tag)
                    return
        except:
            return

while True:
    print('Ready! Waiting for a hashtag from redis...')
        
    tag = r.brpop('archive:tagsin')[1] # add rootTag
    print(f'start: {tag}')
    r.srem('set:tags:archive', tag)
    if r.sismember('scrape:complete', tag):
        continue
    p = Process(target=archive_scrape, args=(tag, ))
    p.start()

# TODO: IF ARCHIVAL SCAN DON'T STOP
# TODO: Fan out search

# TODO: RACE Proxies
# TODO: Find related tags results
# TODO: CREATE AN INDEX TAG > POST
# TODO: CHECK IF ID EXIST AND STOP SCRAPING
# TODO: Dump page to redisjson