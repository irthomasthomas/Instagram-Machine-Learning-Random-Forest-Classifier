from multiprocessing import Process

def getproxies():
    import proxymanager.gotproxies as proxies
    proxies.main()

def scraper():
    import scraper
    scraper.main()

def cachemaker():
    import redis_cache_maker as cache
    cache.main()

def archiver():
    import scraper_bg_archiver
    scraper_bg_archiver.main()

def burst():
    import burst_scraper
    burst_scraper.main()

if __name__ == "__main__":
    print("hello")
    # Process(target=getproxies).start()
    Process(target=scraper).start()
    Process(target=cachemaker).start()
    Process(target=archiver).start()
    Process(target=burst).start()