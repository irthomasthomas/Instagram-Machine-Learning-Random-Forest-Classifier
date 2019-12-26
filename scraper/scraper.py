from InstaLoaderTommy import InstaloaderTommy
import redis
r = redis.Redis(decode_responses=True)

scraper = InstaloaderTommy()
total = 0
while True:
    print('Ready! Waiting for hashtag...')
    tag = r.brpop('tagsin')[1]
    print(f'Received request for {tag}')
    with scraper:
        for count in scraper.get_hashtag_posts(
            tag, resume=True):
            total += count
            print(f'scraped: {total}')
            if total > 170:
                break
