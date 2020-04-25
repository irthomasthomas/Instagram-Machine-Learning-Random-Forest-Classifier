import random
from multiprocessing.dummy import Pool as ThreadPool 
from requests import get, Session
from datetime import datetime
import json
from redis import Redis
from time import perf_counter, sleep
# from proxybroker import Broker
# import asyncio

# TODO: RUN ON LOOP
rdb = Redis()

def check_proxy(proxy):
    session = Session()
    proxies = {
        'https': proxy
    }
    url = 'https://instagram.com'  
    # if random.randint(1, 2) == 1: url = 'https://api.ipify.org'    
    # else: url = 'https://www.w3.org'
    
    try:
        start = perf_counter()
        session.get(url,proxies=proxies,timeout=3, allow_redirects=False)
        end = perf_counter()
        timing = end - start
        rdb.zadd('proxies:', {proxy: timing})
        return proxy
    except:
        return

def getproxies1():
    # LIST 1
    with open("proxy.list","w") as jf:
        jf.write(get("https://raw.githubusercontent.com/fate0/proxylist/master/proxy.list").text)
    jsonlist = []
    with open("proxy.list","r") as jf:
        for x in jf:
            jsonlist.append(json.loads(x))

    proxylist = []
    for p in jsonlist:
        ip = str(p['host'])+":"+str(p['port'])
        if ip not in proxylist:
            proxylist.append(ip)
    print("LIST 1: " + str(len(proxylist)))
    
    # LIST 2
    prsc = json.loads(get("https://raw.githubusercontent.com/sunny9577/proxy-scraper/master/proxies.json").text)
    sourcelist = []
    for i in prsc:
        sourcelist.append(i)
    sourcelist.remove("lastUpdated")
    for source in sourcelist:
        for i in prsc[source]:
            ip =str(i["ip"])+":"+str(i['port'])
            if ip not in proxylist:
                proxylist.append(ip)
    print("LIST 2: " + str(len(proxylist)))
    # LIST 3
    return proxylist

def getproxies2():
    proxylist = []
    for line in get("https://raw.githubusercontent.com/clarketm/proxy-list/master/proxy-list.txt").content.splitlines(True):
        p = line.decode().rstrip().split(" ")[0].split(":")
        try:
            ip = str(p[0])+":"+str(p[1])
            if ip not in proxylist:
                proxylist.append(ip)
        except:
            continue
    print("LIST 3: " + str(len(proxylist)))
    # LIST 4
    for line in get("https://raw.githubusercontent.com/TheSpeedX/SOCKS-List/master/socks.txt").content.splitlines(True):
        try:
            p = line.decode().rstrip().split(" ")[0].split(":")
            ip = str(p[0])+":"+str(p[1])
            if ip not in proxylist:
                proxylist.append(ip)
        except:
            continue
    print("LIST 4: " + str(len(proxylist)))
    return proxylist


def test_proxy_list(proxylist):
    goodproxies = []
    pool = ThreadPool(len(proxylist))

    goodproxies = [x for x in pool.map(check_proxy, proxylist) if x is not None]
    pool.close()
    pool.join()
    print(f'found {len(goodproxies)} proxies')

def main():
    while True:    
        start_time = datetime.now()
        
        print('Getting github proxies...')

        proxylist1 = getproxies1()
        test_proxy_list(proxylist1)
        sleep(300)
        proxylist2 = getproxies2()
        test_proxy_list(proxylist2)
        
        endtime = datetime.now()
        print("Time: " +str(endtime - start_time))
        # proxies = asyncio.Queue()
        # broker = Broker(proxies)
        # tasks = asyncio.gather(
        #     broker.find(types=['HTTPS'], limit=10), get_prox(proxies)
        # )    
        sleep(300)
    

if __name__ == "__main__":
    main()

# with open("pythonproxies.json","w") as file:
#         json.dump(proxylist, file)

# lap2 = datetime.now() - start_time
# # result = ("PYTHON(REQUESTS): "+str(datetime.now())+" Tested " + str(len(proxylist)) + " Found: " + str(len(goodproxies)) + " good proxies in " + str(lap2))
# # print(result)

# with open("results.txt","w") as file:
#         file.write(result)