import lzma
import redis

from contextlib import contextmanager

from instaloader import Instaloader
from instaloader import InstaloaderContext, Post
from instaloader.exceptions import *
import os
from datetime import datetime
from typing import Any, Callable, Dict, Tuple, Iterator, List, Optional, Union
from instaloader.structures import PostComment, PostCommentAnswer, Profile

import random 
import requests
import requests.utils
import json
import time
from DB import Database
import re
import sys
# import ujson
from prometheus_client import Counter, start_http_server

from multiprocessing import Process, Queue, Manager, Pool
from glob import glob
import pathlib

def default_user_agent() -> str:
    return 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' \
           '(KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36'

class InstaloaderTommy(Instaloader):
    def __init__(self):
        super().__init__('Instaloader')
        self.context = InstaloaderContextTommy()
        self.download_comments=True

    def update_comments(self, filename: str, post: Post) -> None:
        print("NEW update_comments")
        def _postcommentanswer_asdict(comment,parentId):
            print("NEW _postcommentanswer")
            # update_comments_db(comment,post,parentId)
            return {'id': comment.id,
                    'created_at': int(comment.created_at_utc.replace(tzinfo=timezone.utc).timestamp()),
                    'text': comment.text,
                    'owner': comment.owner._asdict()}

        def _postcomment_asdict(comment):
            print("NEW _postcomment")
            return {**_postcommentanswer_asdict(comment,comment.id),
                    'answers': sorted([_postcommentanswer_asdict(answer,comment.id) for answer in comment.answers],
                                        key=lambda t: int(t['id']),
                                        reverse=True)}

        def get_unique_comments(comments, combine_answers=False):
            if not comments:
                return list()
            comments_list = sorted(sorted(list(comments), key=lambda t: int(t['id'])),
                                    key=lambda t: int(t['created_at']), reverse=True)
            unique_comments_list = [comments_list[0]]
            for x, y in zip(comments_list[:-1], comments_list[1:]):
                if x['id'] != y['id']:
                    unique_comments_list.append(y)
                elif combine_answers:
                    combined_answers = unique_comments_list[-1].get('answers') or list()
                    if 'answers' in y:
                        combined_answers.extend(y['answers'])
                    unique_comments_list[-1]['answers'] = get_unique_comments(combined_answers)
            return unique_comments_list
        filename += '_comments.json'
        try:
            with open(filename) as fp:
                comments = json.load(fp)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            comments = list()
        comments.extend(_postcomment_asdict(comment) for comment in post.get_comments())
        if comments:
            comments = get_unique_comments(comments, combine_answers=True)
            answer_ids = set(int(answer['id']) for comment in comments for answer in comment.get('answers', []))
            with open(filename, 'w') as file:
                file.write(json.dumps(list(filter(lambda t: int(t['id']) not in answer_ids, comments)), indent=4))
            self.context.log('comments', end=' ', flush=True)

    def get_hashtag_posts(self, hashtag: str, resume=False, end_cursor=False) -> Iterator[Post]:
        """Get Posts associated with a #hashtag."""
        print("get_hashtag_posts")
        has_next_page = True
        if resume:
            end_cursor = get_end_cursor(hashtag)
        # print(str(end_cursor))
        while has_next_page:
            if end_cursor:
                params = {'__a': 1, 'max_id': end_cursor}
            else:
                params = {'__a': 1}
            hashtag_data = self.context.get_json('explore/tags/{0}/'.format(hashtag),
                                                params)['graphql']['hashtag']['edge_hashtag_to_media']
            end_cursor = hashtag_data['page_info']['end_cursor']
            dump_page_json(end_cursor,hashtag_data,hashtag)
            # print(str(len(hashtag_data['edges'])))
            #yield from (Post(self.context, edge['node']) for edge in hashtag_data['edges'])
            yield len(hashtag_data['edges'])
            has_next_page = hashtag_data['page_info']['has_next_page']
            if end_cursor and has_next_page:
                print(bcolors.OKBLUE + "has next page" + bcolors.ENDC)
                # print("end_cursor: "+str(end_cursor))
                update_end_cursor(end_cursor,hashtag, has_next_page)
            else:
                update_end_cursor("",hashtag,0)
    
class InstaloaderContextTommy(InstaloaderContext):
    default_user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    # Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36
    # Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36
    # Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0

    _session = ""

    proxies = []
    @staticmethod
    def fillproxies():
        print("fillproxies")
        with open("goodproxies.json","r") as file:   
            InstaloaderContextTommy.proxies = json.load(file)
    

    def __init__(self, sleep: bool = True, quiet: bool = False, user_agent: Optional[str] = None,
                 max_connection_attempts: int = 3):
        if not InstaloaderContextTommy.proxies:
            InstaloaderContextTommy.fillproxies()
        
        self.user_agent = user_agent if user_agent is not None else default_user_agent()
        if not InstaloaderContextTommy._session:
            InstaloaderContextTommy._session = self.get_anonymous_session()

        # self._session = self.get_anonymous_session()
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
       
    def proxy(self, proxiesList):
        sess = session if session else self._session
        while True:
            try:
                proxy = random.choice(proxiesList)
                proxy = proxy[0] + ":" + proxy[1]
                proxies = {
                        "http" : proxy,
                        "https" : proxy
                        }
                # print(str(proxies))
                ip = sess.get('https://api.ipify.org',proxies=proxies,timeout=3)
                if ip.status_code == 200:
                    break
            except:
                continue
        print('Proxy IP is:'+str(ip.text))
        return proxies

    def get_json(self, path: str, params: Dict[str, Any], host: str = 'www.instagram.com',
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
        # proxies = self.proxy(self.proxies)  
        # ip = sess.get('https://api.ipify.org').text
        try:
            print("my: get_json")
            self.do_sleep()
            for attempt_no in range(100):
                proxies = random.choice(self.proxies)
                print("attempt: " + str(attempt_no))
                print(str(proxies))      
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
                if resp.status_code == 404:
                    print(bcolors.WARNING + "error 404" + bcolors.ENDC)
                    raise QueryReturnedNotFoundException("404 Not Found")
                     
                if resp.status_code == 429:
                    print("TEST error 429: too many requests")
                    time.sleep(3)
                if resp.status_code != 200:
                    print("status code: " + str(resp.status_code))
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
                        continue
                        # raise ConnectionException("Returned \"{}\" status, message \"{}\".".format(resp_json['status'],
                                                                                                # resp_json['message']))
                    else:
                        continue
                        # raise ConnectionException("Returned \"{}\" status.".format(resp_json['status']))
                return resp_json
                    
        except QueryReturnedNotFoundException as err:
            raise err

        except (ConnectionException, json.decoder.JSONDecodeError, requests.exceptions.RequestException) as err:           
            error_string = "JSON Query to {}: {}".format(path, err)
            if _attempt == self.max_connection_attempts:
                proxies = self.proxy(self.proxies)
                # raise ConnectionException(error_string) from err
            self.error(error_string + " [retrying; skip with ^C]", repeat_at_end=False)
            try:
                if is_graphql_query and isinstance(err, TooManyRequestsException):
                    self._ratecontrol_graphql_query(params['query_hash'], untracked_queries=True)
                if is_iphone_query and isinstance(err, TooManyRequestsException):
                    self._ratecontrol_graphql_query('iphone', untracked_queries=True)
                return self.get_json(path=path, params=params, host=host, session=sess, _attempt=_attempt + 1)
            except KeyboardInterrupt:
                self.error("[skipped by user]", repeat_at_end=False)
                raise ConnectionException(error_string) from err
    
class TPost(Post):
    def __init__(self, context: InstaloaderContext, node: Dict[str, Any],
                 owner_profile: Optional['Profile'] = None):
        assert 'shortcode' in node or 'code' in node

        self._context = InstaloaderContextTommy()
        self._node = node
        self._owner_profile = owner_profile
        self._full_metadata_dict = None  # type: Optional[Dict[str, Any]]
        self._rhx_gis_str = None         # type: Optional[str]
        self._location = None            # type: Optional[PostLocation]

        self.comments_list = []

    @property
    def ownerid(self) -> int:
        """The ID of the post owner."""
        return self._node["owner"]["id"]
        
    @property
    def comment_count(self):
        return self._node["edge_media_to_comment"]["count"]

    def get_comments(self) -> Iterator[PostComment]:
        r"""Iterate over all comments of the post.

        Each comment is represented by a PostComment namedtuple with fields text (string), created_at (datetime),
        id (int), owner (:class:`Profile`) and answers (:class:`~typing.Iterator`\ [:class:`PostCommentAnswer`])
        if available.
        """
        print("My: get_comments")
        def _postcommentanswer(node):
            return PostCommentAnswer(id=int(node['id']),
                                     created_at_utc=datetime.utcfromtimestamp(node['created_at']),
                                     text=node['text'],
                                     owner=Profile(self._context, node['owner']))

        def _postcommentanswers(node):
            if 'edge_threaded_comments' not in node:
                return
            answer_count = node['edge_threaded_comments']['count']
            if answer_count == 0:
                # Avoid doing additional requests if there are no comment answers
                return
            answer_edges = node['edge_threaded_comments']['edges']
            if answer_count == len(answer_edges):
                # If the answer's metadata already contains all comments, don't do GraphQL requests to obtain them
                yield from (_postcommentanswer(comment['node']) for comment in answer_edges)
                return
            yield from (_postcommentanswer(answer_node) for answer_node in
                        self._context.graphql_node_list("51fdd02b67508306ad4484ff574a0b62",
                                                        {'comment_id': node['id']},
                                                        'https://www.instagram.com/p/' + self.shortcode + '/',
                                                        lambda d: d['data']['comment']['edge_threaded_comments']))

        def _postcomment(node):
            return PostComment(*_postcommentanswer(node),
                               answers=_postcommentanswers(node))
        
        if self.comments < 4:
            # Avoid doing additional requests if there are no comments
            return
        try:
            comment_edges = self._field('edge_media_to_parent_comment', 'edges')
            answers_count = sum([edge['node']['edge_threaded_comments']['count'] for edge in comment_edges])
            threaded_comments_available = True
        except KeyError:
            comment_edges = self._field('edge_media_to_comment', 'edges')
            answers_count = 0
            threaded_comments_available = False

        if self.comments == len(comment_edges) + answers_count:
            # If the Post's metadata already contains all parent comments, don't do GraphQL requests to obtain them
            yield from (_postcomment(comment['node']) for comment in comment_edges)
            return
        yield from (_postcomment(node) for node in
                    self._context.graphql_node_list(
                        "97b41c52301f77ce508f55e66d17620e" if threaded_comments_available
                        else "f0986789a5c5d17c2400faebf16efd0d",
                        {'shortcode': self.shortcode},
                        'https://www.instagram.com/p/' + self.shortcode + '/',
                        lambda d:
                        d['data']['shortcode_media'][
                            'edge_media_to_parent_comment' if threaded_comments_available else 'edge_media_to_comment'],
                        self._rhx_gis))
          
    def _obtain_metadata(self):
        print("my: obtain_metadata")
        # print(self.shortcode)
        if not self._full_metadata_dict:
            pic_json = self._context.get_json("p/{0}/".format(self.shortcode), params={})
            self._full_metadata_dict = pic_json['entry_data']['PostPage'][0]['graphql']['shortcode_media']
            self._rhx_gis_str = pic_json.get('rhx_gis')
            if self.shortcode != self._full_metadata_dict['shortcode']:
                self._node.update(self._full_metadata_dict)
                raise PostChangedException

def update_end_cursor(end_cursor,hashtag,has_next_page):
    inputs = (end_cursor,hashtag,has_next_page,datetime.now())
        
    sql = """ REPLACE INTO scraper_status(end_cursor,hashtag,has_next_page,update_date)
                VALUES(?,?,?,?) """
    with Database("instagram.sqlite3") as db:
        db.execute(sql,inputs)
    
def get_end_cursor(hashtag):
    sql = """ SELECT end_cursor 
            FROM scraper_status 
            WHERE hashtag = ? 
            LIMIT 1 """
    with Database("instagram.sqlite3") as db:
        db.execute(sql,(hashtag,))
        end_cursor = db.fetchone()
    return end_cursor[0]

def dump_page_json(filename,page,dir):
        print("dump_page_json: "+str(filename)+" "+str(dir))
        filename = dir + '/' + filename+".json"
        os.makedirs(dir, exist_ok=True)        
        with open(filename, 'wt') as fp:
            json.dump(page, fp=fp, indent=4, sort_keys=True) 

def get_proxy_db():
    with Database("instagram.sqlite3") as db:
        sql = ("""SELECT ip 
            from proxy limit 20 """)
        db.execute(sql)
        proxieslist = db.fetchall()
    return proxieslist

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_posts_from_dir(path):
    for file in os.scandir(path):
        with open(file, 'r') as fp:
            f = fp.read()
            resp_json = json.loads(f)
            yield from (TPost(scraper.context, edge['node']) for edge in resp_json['edges'])

def load_post_from_file(file):
    with open(file, 'r') as fp:
        f = fp.read()
        resp_json = json.loads(f)
        yield from (TPost(scraper.context, edge['node']) for edge in resp_json['edges'])

def load_posts_file_dir(path):
    for file in os.scandir(path):
        with open(file, 'r') as fp:
            f = fp.read()
            yield json.loads(f)

def load_json_posts_file_dir(path,posts):
    for file in os.scandir(path):
        with open(file, 'r') as fp:
            f = fp.read()
            posts.append(json.loads(f))

def fill_mp(pages,posts):
    for page in pages:
        fill_post_list(page,posts)

def fill_post_list(page,posts):
    # print("fill_post_list: ")
    # print(str(len(posts)))
    # print(str(page))
    for edge in page['edges']:
        posts.append(TPost(scraper.context, edge['node']))
    # posts.append(TPost(scraper.context, edge['node']) for edge in page['edges'])

def get_coms(post,shared_list):
    count = 0
    try:
        for comment in post.get_comments():
            count += 1
            post.comments_list.append(comment)
            # c.inc(1) 
            # print(str(post.mediaid))
            # print("comment: " + str(comment.text))
            for answer in comment.answers:
                count += 1
                post.comments_list.append(answer)
                # c.inc(1) 
                # print("answer: " + str(answer.text))
        print("comms count" + str(count))
        # print(str(post.comments_list))
        print("comms len: " + str(len(post.comments_list)))
        shared_list.append(count)
    except KeyboardInterrupt:
        print("ctrl c")
        exit(0)
    except Exception as e:
        print("exception " + str(e))
        # continue

def save_to_file(data, filename: str) -> None:
    with lzma.open(filename, 'wt') as fp:
            json.dump(data, fp=fp, separators=(',', ':'))

def extract_hashtags(caption) -> List[str]:
    """List of all lowercased hashtags (without preceeding #) that occur in the Post's caption."""
    hashtags = []
    # if not self.caption:
        # return []
    # This regular expression is from jStassen, adjusted to use Python's \w to support Unicode
    # http://blog.jstassen.com/2016/03/code-regex-for-instagram-username-and-hashtags/
    hashtag_regex = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
    hashtags = re.findall(hashtag_regex, caption.lower())
    return hashtags

def post_dict(edge):
        try:
            likes = edge['node']['edge_liked_by']['count']
        except:
            likes = ""
        try:
            comments = edge['node']['edge_media_to_comment']['count']
        except:
            comments = ""
        try:
            caption = edge['node']['edge_media_to_caption']['edges'][0]['node']['text']
        except:
            caption = ""
        try:
            typename = edge['node']['__typename']
        except:
            typename = ""
        try:
            owner_id = edge['node']['owner']['id']
        except:
            owner_id = ""
        try:
            shortcode = edge['node']['shortcode']
        except:
            shortcode = ""
        try:
            timestamp = edge['node']['taken_at_timestamp']
        except:
            timestamp = ""
        scrape_date = time.time()
        return {"likes": likes, "comments":comments,"caption":caption, 
        "typename":typename,"owner_id":owner_id,"shortcode":shortcode,"timestamp":timestamp,"scrape_date":scrape_date}
    
def starts(path):
    pipe = r.pipeline()
    for page in load_posts_file_dir(path):
        for edge in page['edges']:
            post = post_dict(edge)
            try:
                id = edge['node']['id']
                pipe.hmset("post:"+str(id),post)
                hashtags = extract_hashtags(post['caption'])
                # pipe.lpush("tags:"+str(id),*hashtags)
                # if not r.exists("tags:"+str(id)):
                r.sadd("posttags:"+str(id),*hashtags)
                # if not r.exists("tags:"+str(id)):            
                for tag in hashtags:
                    pipe.sadd("tagposts:"+tag,id)
            except:
                continue
            # id = ""
        # print("tm to fill pipe: " + str(time.perf_counter() - start))
    pipe.execute()
   
start = time.perf_counter()

session = requests.session()
scraper = InstaloaderTommy()
commentCount = Counter('scraped_comments', 'Session Scraped Comments')
hashtagPostCount = Counter('scraped_hashtag_posts', 'Session scraped hashtag posts')
start_http_server(8080)

if len(sys.argv) >= 1:
     path=sys.argv[1]
#     start_tag = sys.argv[2]

""" 
with Manager() as manager:
    posts = manager.list()
    # processes = []
    for page in pages:
        p = Process(target=fill_post_list, args=(page,posts))
        p.start()
        # processes.append(p)
    # for p in processes:
    p.join()
    print("posts: " + str(len(posts)))
 """
r = redis.Redis(host='localhost', port=6379, db=0)
redis_key = "posts"
processes = []

subdirs = pathlib.Path(path).glob('*/')
for i in subdirs:
    if os.path.isdir(i):
        p = Process(target=starts,args=(i,))
        processes.append(p)
        p.start()
for proc in processes:
    proc.join()

posts2 = []
# starts(path)
        # if not r.exists(id):
end = time.perf_counter()
print("tm to exc pipe: " + str(end - start))


# start = time.clock()
# pool = Pool(50)
# pipe = r.pipeline()
# for page in load_posts_file_dir(path):
#     p = Process(target=set_pipe,args=(page,pipe)) # goodproxies = [x for x in pool.map(check_proxy, proxylist) if x is not None]
#     p.start()
#     p.join()
#     # set_pipe(page, pipe)
# print(time.clock() - start)

# pipe.execute()
# print(time.clock() - start)

    # r.lpush(redis_key, json.dump(post))

# yield from (TPost(scraper.context, edge['node']) for edge in resp_json['edges'])

""" 
processes = []
pool = Pool(50)
manager = Manager()
shared_list = manager.list()
[pool.apply_async(get_coms, args=[post,shared_list]) for post in posts2 ]
pool.close()
pool.join()
print("shared_list:")
print(shared_list)
"""
# pool.map(get_coms, posts2,shared_list)  

# for post in shared_list:
    # print(post.mediaid)
    # print(str(len(post.comments_list)))
    # proc = Process(target=get_coms,args=(post,))
    # processes.append(proc)
    # proc.start()

# for proc in processes:
    # proc.join()

"""
# with scraper:
    #     for count in scraper.get_hashtag_posts("headyart",resume=True):
    #         print("COUNT")
    #         print(str(count))

    # pool = ThreadPool(len(proxylist))
    # goodproxies = [x for x in pool.map(check_proxy, proxylist) if x is not None]
 
# yield from (TPost(scraper.context, edge['node']) for edge in resp_json['edges'])

"""
"""

if os.path.isdir(path):
    print(str(path))
    try:
        for post in load_posts_from_dir(path):
            # print(post.caption)
            try:
                for comment in post.get_comments():
                    # c.inc(1) 
                    # print(str(post.mediaid))
                    print("comment: " + str(comment.text))
                    for answer in comment.answers:
                        # c.inc(1) 
                        print("answer: " + str(answer.text))
            except KeyboardInterrupt:
                print("ctrl c")
                exit(0)
            except e:
                print(str(e))
                continue
    except KeyboardInterrupt:
        print("ctrl c")
        exit(0)
           
 """