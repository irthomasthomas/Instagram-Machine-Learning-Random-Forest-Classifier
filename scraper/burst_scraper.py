from InstaLoaderTommy import InstaloaderTommy
import redis
import time
from multiprocessing import Process
import random

r = redis.Redis(decode_responses=True)

scraper = InstaloaderTommy(mode='burst')


def scrape(second_tag, root_tag, num_to_scrape):
    # r.delete(f'scrape:finished:{tag}')
    print(f'scrape {second_tag} {root_tag}')
    with scraper:
        total = 0
        try:
            for count in scraper.get_hashtag_posts(
                    hashtag=second_tag, is_root_tag=root_tag,
                    dump_page=True, resume=False,
                    duplicate_check=False, related_burst=True):
                total += count
                # Trigger cache generator
                print(f'scraped: {total}')
                
                # added = r.sadd('set:cache:queue', tag)
                # if added:
                #     r.lpush('cache:queue:ready', tag)
                
                if total > num_to_scrape:
                    print('FINISHED')
                    # trigger archive background scraper
                    # scrape_more = r.sadd('set:tags:archive:queue', tag)
                    # print(f'scrape_more: {scrape_more}')
                    # if scrape_more:
                        # r.lpush('archive:tagsin', tag)
                    total = 0
                    # Remove from burst queue set
                    r.srem('queue:burst', second_tag)
                    return
        except:
            return

def main():         
    while True:

        num_to_scrape = 50

        print('Ready! Waiting for a hashtag from redis...')
        # TODO:DONE:bg_scraper.py background scraping. pop a tag and resume page and total
        # TODO: HOW TO BENCHMARK ON VARIABLE SPEED CPU

        # tags = r.brpop('list:burst')[1] # add rootTag tuple
        second_tag = r.brpop('list:burst')[1] # add rootTag tuple
        # if random.randint(1,2) == 2:
        #     time.sleep(0.4)
        root_tag = r.get(f'root:tag:{second_tag}')
        
        # r.delete(f'root:tag:{second_tag}')

        print(f'BURST REQ: root:{root_tag} second:{second_tag} ')
        
        # 
        # scraped = r.get(f'scraped:recent:{tag}')
        # # print(f'scraped_tag: {scraped}')
        # if scraped:
        #     print(f'FOUND SCRAPED_RECENT KEY')
        #     print(f'ABORTING')
        #     continue
        # else:
        #     r.set(key, "True")
        p = Process(target=scrape, args=(second_tag, root_tag, num_to_scrape))
        p.start()