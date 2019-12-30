from InstaLoaderTommy import InstaloaderTommy
import redis
import time
from multiprocessing import Process

r = redis.Redis(decode_responses=True)

scraper = InstaloaderTommy()


def scrape(tag):
    with scraper:
        total = 0
        for count in scraper.get_hashtag_posts(
            tag, resume=True):
            total += count
            print(f'scraped: {total}')
            if total > 1000:
                total = 0
                break

while True:
    print('Ready! Waiting for a hashtag from redis...')
    tag = r.brpop('tagsin')[1]
    print(f'Received request for {tag}')
    start = time.perf_counter()
    p = Process(target=scrape, args=(tag,))
    p.start()
    end = time.perf_counter()
    print(f'TIMER: {end - start}')


    
