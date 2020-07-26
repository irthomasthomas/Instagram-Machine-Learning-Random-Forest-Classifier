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
    # Process(target=getproxies).start()
    Process(target=scraper, daemon=True).start()
    Process(target=cachemaker, daemon=True).start()
    Process(target=archiver, daemon=True).start()
    Process(target=burst, daemon=True).start()