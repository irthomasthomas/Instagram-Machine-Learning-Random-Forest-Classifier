from InstaLoaderTommy import InstaloaderTommy
import redis
import time
from multiprocessing import Process

r = redis.Redis(decode_responses=True)

scraper = InstaloaderTommy()
# TODO: DECIDE RELEVANCE BASED ON 1 PAGE SCRAPE AND RELEVANT
# IF NOT RELEVANT: msg: Sorry this tag doesn't have much stuff..
# ... here is some related stuff...
# return related tags

# TODO: RELATED RESULTS
# TODO: CHECK TIME SINCE SCRAPE

# TODO: ADD ALL BLOOM / CUCKOO FILTER
# TODO: CHECK HASHTAG/POSTID LIST AND STOP ON DUPLICATE
# TODO: CHECK AGAINST GLOBAL LIST AND RETRIEVE PREDICTION
# TODO: GEAR CHECK FOR EXISTING PREDICTION

def scrape(tag, num_to_scrape):
    # TODO: NAMESPACE so I can do eg TODO:redis:BigOFunc
    r.delete(f'scrape:finished:{tag}')
    with scraper:
        total = 0
        # TODO: check dump
        try:
            for count in scraper.get_hashtag_posts(
                    tag, dump_page=True, resume=False,
                    duplicate_check=True):
                # TODO:Should I control scraping here or in class?
                total += count
                # Trigger cache generator
                print(f'scraped: {total}')
                
                if total > num_to_scrape:
                    added = r.sadd('set:cache:queue', tag)
                    if added:
                        r.lpush('cache:queue:ready', tag)
                    print('trigger archive background scraper')
                    scrape_more = r.sadd('tags:archive:queue', tag)
                    print(f'scrape_more: {scrape_more}')
                    if scrape_more:
                        r.lpush('archive:tagsin', tag)
                    total = 0
                    # TODO: DONE: ONLY CACHE READY WHEN STREAM OUT > 10
                    # TODO: DONE: ADD TO SET IF SUCCESS LPUSH
                    return
        except:
            return
# TODO: REWRITE IN CRYSTAL
# TODO: STOP SCRAPING WHEN DUPLICATE FOUND
# CUCKOO FILTER?
# store every prediction. if exists retrieve prediction else predict.
# TODO: VARY SCRAPING BY SIZE AND CONTENT OF HASHTAG FEED

while True:

    num_to_scrape = 1800

    print('Ready! Waiting for a hashtag from redis...')
    # TODO: background scraping. pop a tag and resume page and total
    # TODO: SUBSCRIBE AND BARK
    # TODO: BACKGROUND SCRAPE HISTORIC    

    tag = r.brpop('list:tagsin')[1] # add rootTag
    r.srem('tagsin', tag)
    print(f'SCRAPE REQ: {tag}')

    scraped = r.get(f'scraped:recent:{tag}')
    # print(f'scraped_tag: {scraped}')
    if scraped:
        print(f'FOUND SCRAPED_RECENT KEY')
        print(f'ABORTING')
        continue
    else:
        r.set(f'scraped:recent:{tag}', "True")
    p = Process(target=scrape, args=(tag, num_to_scrape))
    p.start()
    r.expire(f'scraped:recent:{tag}', 600)

# TODO: DONE - IF ARCHIVAL SCAN DON'T STOP
# TODO: DONE - Fan out search

# TODO: RACE Proxies
# TODO: Find related tags results
# TODO: CREATE AN INDEX TAG > POST
# TODO: CHECK IF ID EXIST AND STOP SCRAPING
# TODO: Dump page to redisjson