import random
from multiprocessing.dummy import Pool as ThreadPool 
from requests import get, Session
from datetime import datetime
import json
from proxybroker import Broker, ProxyPool
from proxybroker.errors import NoProxyError
import asyncio
import time
import redis

async def show(proxies):
    session = Session()
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    test_url = 'https://www.instagram.com/'
    while True:
        proxy = await proxies.get()
        if proxy is None: break
        ip = str(proxy.host) + ":" + str(proxy.port)
        myProxy = {'https': ip}
        try:
            start = time.perf_counter()
            session.get(
                test_url,
                proxies=myProxy,
                timeout=3,
                allow_redirects=False)
            end = time.perf_counter()
            timing = end - start
            print(time.localtime)
            print(f'{proxy} time: {timing}')
            result = r.zadd('proxies:', {ip: timing})
            print(result)
            print(time.strftime('%X %x %Z'))

        except:
            print(time.strftime('%X %x %Z'))
            print(f'Rejected: {proxy}')
            # result = r.zrem('proxies:', ip)
            # print(result)
            
            continue
        # print(f'Found proxy: {proxy}')
        # zpopmin / zrem


def main():
    proxies = asyncio.Queue()
    broker = Broker(proxies, timeout=3)
    tasks = asyncio.gather(
        broker.find(types=['HTTPS', 'HTTP']),
        show(proxies))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(tasks)


if __name__ == '__main__':
    main()
