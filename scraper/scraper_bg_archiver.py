from InstaLoaderTommy import InstaloaderTommy
import redis
import time
from multiprocessing import Process

r = redis.Redis(decode_responses=True)

scraper = InstaloaderTommy()


def monitor(start, posts):
    
    print(f'start: {start}')
    end = time.perf_counter()
    timing = end - start
    print(f'timing: {timing}')
    start = time.perf_counter()
                
    print(f'posts: {posts}')
    post_per_sec = posts / timing
    print(f'post /sec {post_per_sec}')

    # TODO: performance stream
    r.xadd('stats:perf:archive_scrape',
        {"timing": timing, "num_scraped": posts},
        maxlen=5000)


def archive_scrape(tag):
    # TODO: TIMING SET START KEY AND END KEY
    # DON'T STORE KEYS. STORE IN HLL OR OTHER PDS
    # timing stream
    # x[module: archive_scrape, start, end, count
    # or pps
    # or log start and log end
    start = time.perf_counter()
    print(f'archive scraper {tag}')
    num_to_scrape = 300
    with scraper:
        total = 0
        try:
            for count in scraper.get_hashtag_posts(
                    tag, dump_page=True, resume=True,
                    duplicate_check=False, archive_run=True):
                monitor(start, count)
                start = time.perf_counter()
                total += count
                print(f'scraped total: {total}')
                added = r.sadd('set:cache:queue', tag)
                if added:
                    r.lpush('cache:queue:ready', tag)
                if total > num_to_scrape:
                    print('SLEEP')
                    time.sleep(1)
                    total = 0
                    not_exists = r.sadd('tags:archive:queue', tag)
                    print(f'not_exist: {not_exists}')
                    if not_exists:
                        print('push to tagsin')
                        r.lpush('archive:tagsin', tag)
                    return
        except:
            return

while True:
    print('Archiver Ready! Waiting for a hashtag from redis...')
    
    tag = r.brpop('archive:tagsin')[1] # add rootTag
    print(f'start: {tag}')
    r.srem('tags:archive:queue', tag)
    if r.exists(f'scrape:complete:{tag}'):
        print(f'archiving already complete:{tag}')
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