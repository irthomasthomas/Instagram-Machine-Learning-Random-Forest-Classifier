from contextlib import contextmanager

from instaloader import Instaloader
from instaloader import InstaloaderContext, Post
from instaloader.exceptions import *
from instaloader.structures import (Highlight, JsonExportable, Post, PostLocation, Profile, Story, StoryItem,
                         save_structure_to_file, load_structure_from_file, PostComment, PostCommentAnswer)
import os
import sys
from datetime import datetime
from typing import Any, Callable, Dict, Tuple, Iterator, List, Optional, Union

import random 
import requests
import requests.utils
import json
import time
from DB import Database
import re

def default_user_agent() -> str:
    return 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' \
           '(KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36'

class InstaloaderTommy(Instaloader):
    def __init__(self):
        super().__init__('Instaloader')
        self.context = InstaloaderContextTommy()
    
    def get_hashtag_posts(self, hashtag: str, proxies, resume=False, end_cursor=False) -> Iterator[Post]:
        """Get Posts associated with a #hashtag."""
        has_next_page = True
        if resume:
            end_cursor = get_end_cursor(hashtag)

        while has_next_page:
            if end_cursor:
                params = {'__a': 1, 'max_id': end_cursor}
            else:
                params = {'__a': 1}
                
            hashtag_data = self.context.get_json('explore/tags/{0}/'.format(hashtag),
                                                 proxies, params)['graphql']['hashtag']['edge_hashtag_to_media']
            end_cursor = hashtag_data['page_info']['end_cursor']
            dump_page_json(end_cursor,hashtag_data,hashtag)
            
            #yield from (Post(self.context, edge['node']) for edge in hashtag_data['edges'])
            has_next_page = hashtag_data['page_info']['has_next_page']
            if end_cursor and has_next_page:
                print(bcolors.OKBLUE + "has next page" + bcolors.ENDC)
                print("end_cursor: "+str(end_cursor))
                update_end_cursor(end_cursor,hashtag, has_next_page)
            else:
                update_end_cursor("",hashtag,0)
    
    """ 
    def download_pic(self, filename: str, url: str, mtime: datetime, filename_suffix: Optional[str] = None) -> bool:
        return True
        
        def download_profilepic(self, name: str, url: str) -> None:
            if download_profilepic:
                super().download_profilepic(name, url)
        
        def check_profile_id(self, profile: str, profile_metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
            if track_profile_id:
                return super().check_profile_id(profile, profile_metadata)
            return profile, 0
        """
    
    @contextmanager
    def anonymous_copy(self):
        new_loader = InstaloaderTommy(self.sleep, self.quiet, self.user_agent,
                                             self.dirname_pattern, self.filename_pattern,
                                             self.download_videos, self.download_geotags,
                                             self.save_captions, self.download_comments,
                                             self.save_metadata, self.max_connection_attempts)
        new_loader.previous_queries = self.previous_queries
        yield new_loader
        self.error_log.extend(new_loader.error_log)
        self.previous_queries = new_loader.previous_queries
 
class InstaloaderContextTommy(InstaloaderContext):
    default_user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'

    def __init__(self, sleep: bool = True, quiet: bool = False, user_agent: Optional[str] = None,
                 max_connection_attempts: int = 3):

        self.user_agent = user_agent if user_agent is not None else default_user_agent()
        self._session = self.get_anonymous_session()
        self.username = None
        self.sleep = sleep
        self.quiet = quiet
        self.max_connection_attempts = max_connection_attempts
        self._graphql_page_length = 50
        self._root_rhx_gis = None
        self.two_factor_auth_pending = None

        self.error_log = []                      # type: List[str]

        # For the adaption of sleep intervals (rate control)
        self._graphql_query_timestamps = dict()  # type: Dict[str, List[float]]
        self._graphql_earliest_next_request_time = 0.0

        # Can be set to True for testing, disables supression of InstaloaderContext._error_catcher
        self.raise_all_errors = False

        # Cache profile from id (mapping from id to Profile)
        self.profile_id_cache = dict()   

    def get_json(self, path: str, proxiesList, params: Dict[str, Any], host: str = 'www.instagram.com',
                 session: Optional[requests.Session] = None, _attempt=1) -> Dict[str, Any]:
        """JSON request to Instagram.

        :param path: URL, relative to the given domain which defaults to www.instagram.com/
        :param params: GET parameters
        :param host: Domain part of the URL from where to download the requested JSON; defaults to www.instagram.com
        :param session: Session to use, or None to use self.session
        :return: Decoded response dictionary
        :raises QueryReturnedBadRequestException: When the server responds with a 400.
        :raises QueryReturnedNotFoundException: When the server responds with a 404.
        :raises ConnectionException: When query repeatedly failed.
        """
        is_graphql_query = 'query_hash' in params and 'graphql/query' in path
        is_iphone_query = host == 'i.instagram.com'
        sess = session if session else self._session
        proxies = proxy(proxiesList)
        time.sleep(0)
        ip = sess.get('https://api.ipify.org').text
        print(str(proxies))        
        try:
            print("Trying")
            self.do_sleep()
            for attempt_no in range(100):
                resp = sess.get('https://{0}/{1}'.format(host, path), 
                    proxies=proxies, params=params, allow_redirects=True, timeout=3)
                
                while resp.is_redirect:
                    redirect_url = resp.headers['location']
                    self.log('\nHTTP redirect from https://{0}/{1} to {2}'.format(host, path, redirect_url))
                    if redirect_url.startswith('https://{}/'.format(host)):
                        resp = sess.get(redirect_url if redirect_url.endswith('/') else redirect_url + '/',
                                        proxies=proxies, params=params, allow_redirects=False, timeout=4)
                    else:
                        break
                if resp.status_code == 400:
                    print(bcolors.WARNING + "error 400" + bcolors.ENDC)
                    # raise QueryReturnedBadRequestException("400 Bad Request")
                if resp.status_code == 404:
                    print(bcolors.WARNING + "error 404" + bcolors.ENDC)
                    # raise QueryReturnedNotFoundException("404 Not Found")
                if resp.status_code == 429:
                    print("error 429: too many requests")
                    # proxies = proxy(proxiesList)
                    #continue
                    #raise 
                    #TooManyRequestsException("429 Too Many Requests")
                if resp.status_code != 200:
                    print("status code: " + str(resp.status_code))
                    proxies = proxy(proxiesList)
                    continue
                    # raise ConnectionException("HTTP error code {}.".format(resp.status_code))
                is_html_query = not is_graphql_query and not "__a" in params and host == "www.instagram.com"
                if is_html_query:
                    match = re.search(r'window\._sharedData = (.*);</script>', resp.text)
                    if match is None:
                        continue
                        # raise ConnectionException("Could not find \"window._sharedData\" in html response.")
                    return json.loads(match.group(1))
                else:
                    resp_json = resp.json()
                if 'status' in resp_json and resp_json['status'] != "ok":
                    if 'message' in resp_json:
                        proxies = proxy(proxiesList)
                        continue
                        # raise ConnectionException("Returned \"{}\" status, message \"{}\".".format(resp_json['status'],
                                                                                                # resp_json['message']))
                    else:
                        proxies = proxy(proxiesList)
                        continue
                        # raise ConnectionException("Returned \"{}\" status.".format(resp_json['status']))
                return resp_json
        except (ConnectionException, json.decoder.JSONDecodeError, requests.exceptions.RequestException) as err:
            error_string = "JSON Query to {}: {}".format(path, err)
            if _attempt == self.max_connection_attempts:
                proxies = proxy(proxiesList)
                # raise ConnectionException(error_string) from err
            self.error(error_string + " [retrying; skip with ^C]", repeat_at_end=False)
            try:
                if is_graphql_query and isinstance(err, TooManyRequestsException):
                    self._ratecontrol_graphql_query(params['query_hash'], untracked_queries=True)
                if is_iphone_query and isinstance(err, TooManyRequestsException):
                    self._ratecontrol_graphql_query('iphone', untracked_queries=True)
                return self.get_json(path=path, proxiesList = proxiesList, params=params, host=host, session=sess, _attempt=_attempt + 1)
            except KeyboardInterrupt:
                self.error("[skipped by user]", repeat_at_end=False)
                raise ConnectionException(error_string) from err
   
    """ 
    def get_raw(self, url: str, _attempt=1) -> requests.Response:
        Downloads a file anonymously.

        :raises QueryReturnedNotFoundException: When the server responds with a 404.
        :raises QueryReturnedForbiddenException: When the server responds with a 403.
        :raises ConnectionException: When download failed.

        .. versionadded:: 4.2.1
        print("GET RAW...")
        time.sleep(4)
        
        
        with self.get_anonymous_session() as anonymous_session:
            resp = anonymous_session.get(url, stream=True, proxies=proxies)
        if resp.status_code == 200:
            resp.raw.decode_content = True
            return resp
        else:
            if resp.status_code == 403:
                # suspected invalid URL signature
                raise QueryReturnedForbiddenException("403 when accessing {}.".format(url))
            if resp.status_code == 404:
                # 404 not worth retrying.
                raise QueryReturnedNotFoundException("404 when accessing {}.".format(url))
            raise ConnectionException("HTTP error code {}.".format(resp.status_code))
    """

def get_proxy_db():
    with Database("instagram.sqlite3") as db:
        sql = ("""SELECT ip 
            from proxy limit 20 """)
        db.execute(sql)
        proxieslist = db.fetchall()
    return proxieslist

def proxy(proxiesList):
    sess = session if session else self._session
    while True:
        try:
            proxy = random.choice(proxiesList)
            proxy = proxy[0] + ":" + proxy[1]
            proxies = {
                    "http" : proxy,
                    "https" : proxy
                    }
            ip = sess.get('https://api.ipify.org',proxies=proxies,timeout=3)
            if ip.status_code == 200:
                break
        except:
            continue
    print('My proxies IP is:'+str(ip.text))
    return proxies

def get_rand_hashtag():
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="@man86ETAfipoz",
            database="ISAMinstagram",
            use_pure=True
            )
        id = random.randrange(1,200,1)
        cur = mydb.cursor(buffered=True)
        sql = ("""SELECT tag FROM hashtag
                WHERE id = ?
                """)
        values = (id,)
        cur.execute(sql,values)
        hashtag = cur.fetchone
        for row in cur:
        #print("cursor "+str(row[0]))
            return row[0]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def load_from_files(path, start_tag):
    for file in os.scandir(path):
        with open(file, 'r') as fp:
            f = fp.read()
            resp_json = json.loads(f)
            yield from (Post(L.context, edge['node']) for edge in resp_json['edges'])
        

"""     for file in os.scandir(path):
        print(str(file))
        with open(file, 'r') as fp:
            print(str(fp))
            resp_json = fp.read.json()
            print(str(resp_json))
            yield from (Post(L.context, edge['node']) for edge in resp_json['edges'])
            post = load_structure_from_file(L.context, file.path)
 """

proxylist = []
with open("goodproxies.json","r") as file:   
    proxylist = json.load(file)

# with Database("instagram.sqlite3") as db:
#     hashtags = db.query('SELECT tag FROM hashtag limit 10')
#     # proxies = db.query('SELECT ip FROM proxy limit 100')
#     print(str(hashtags))
#     # print(str(proxies))

L = InstaloaderTommy()
sql = """ INSERT OR IGNORE INTO post(id, date_utc)
                                VALUES(?,?)
                                """

if len(sys.argv) >= 1:
    path=sys.argv[1]
    start_tag = sys.argv[2]
    
if os.path.isdir(path):
    print(str(path))
    with Database("instagram.sqlite3") as db:
        for post in load_from_files(path,start_tag):
            values = (post.mediaid, post.date_utc)
            r = db.execmany(sql,values)
            print(str(r))
            # print(str(post))
            # print(str(post.caption))
