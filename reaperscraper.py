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
from time import time, sleep
from DB import Database
import re
import sys

from instaphyte import Instagram

from instaphyte.api import InstagramAPI


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

    def get_hashtag_posts(self, hashtag: str, proxies, resume=False, end_cursor=False) -> Iterator[Post]:
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
                                                 proxies, params)['graphql']['hashtag']['edge_hashtag_to_media']
            end_cursor = hashtag_data['page_info']['end_cursor']
            dump_page_json(end_cursor,hashtag_data,hashtag)
            
            #yield from (Post(self.context, edge['node']) for edge in hashtag_data['edges'])
            has_next_page = hashtag_data['page_info']['has_next_page']
            if end_cursor and has_next_page:
                print(bcolors.OKBLUE + "has next page" + bcolors.ENDC)
                # print("end_cursor: "+str(end_cursor))
                update_end_cursor(end_cursor,hashtag, has_next_page)
            else:
                update_end_cursor("",hashtag,0)
    
class InstaloaderContextTommy(InstaloaderContext):
    default_user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    
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
            print("Trying")
            self.do_sleep()
            for attempt_no in range(100):
                proxies = random.choice(self.proxies)
                print(attempt_no)
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
                if resp.status_code == 429:
                    print("error 429: too many requests")
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

 

    def get_comments(self) -> Iterator[PostComment]:
        r"""Iterate over all comments of the post.

        Each comment is represented by a PostComment namedtuple with fields text (string), created_at (datetime),
        id (int), owner (:class:`Profile`) and answers (:class:`~typing.Iterator`\ [:class:`PostCommentAnswer`])
        if available.
        """
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
        
        if self.comments == 0:
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
        print(self.shortcode)
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
            # yield from (TPost(InstaloaderContextTommy, edge['node']) for edge in resp_json['edges'])
            yield from (TPost(scraper.context, edge['node']) for edge in resp_json['edges'])

""" 
class InstagramT(Instagram):
    proxies = []
    @staticmethod
    def fillproxies():
        print("fillproxies")
        with open("goodproxies.json","r") as file:   
            InstagramT.proxies = json.load(file)

    def __init__(self, access_token):
        super().__init__()

        self.access_token = access_token
        self.url = "https://api.instagram.com/v1"
        self.request_rate = 1

        self.last_request = time()
        if not InstagramT.proxies:
            InstagramT.fillproxies()

    def api_call(self, edge, parameters, return_results=True):
        proxies = random.choice(self.proxies)

        parameters['access_token'] = self.access_token
        req = self.get(f"{self.url}/{edge}", params=parameters)
        # req = self.get(f"{self.url}/{edge}", proxies=proxies, params=parameters)

        time_diff = time() - self.last_request
        if time_diff < self.request_rate:
            sleep(time_diff)

        self.last_request = time()

        if return_results:
            return req.json()
    
    
    def endpoint_node_edge(self, endpoint, node, edge=None, params=None):
        parameters = {}
        parameters = self.merge_params(parameters, params)

        if edge:
            return self.api_call(f"{endpoint}/{node}/{edge}", parameters)
        else:
            return self.api_call(f"{endpoint}/{node}", parameters)

    def user(self, user, params=None):
        parameters = {}
        parameters = self.merge_params(parameters, params)

        return self.api_call(f"users/{user}", parameters)

class Instagram(Source):
    def __init__(self):
        super().__init__()

        self.api = InstagramAPI()

    class InstagramIter(Iter):
        def __init__(self, node, function, count, response_key):
            super().__init__()

            self.node = node
            self.function = function
            self.max = count
            self.response_key = response_key
            self.max_id = None

        def get_data(self):
            self.page_count += 1

            try:
                self.response = self.function(self.node, self.max_id)

                page = self.response['graphql'][self.response_key][
                    "edge_" + self.response_key + "_to_media"]
                self.data = page["edges"]
                self.max_id = page["page_info"]["end_cursor"]

                if not self.max_id:
                    raise StopIteration

            except ApiError as e:
                raise IterError(e, vars(self))

    def hashtag(self, tag, count=0):
        return self.InstagramIter(tag, self.api.hashtag, count, "hashtag")

    def location(self, tag, count=0):
        return self.InstagramIter(tag, self.api.location, count, "location")

"""

class NewInstagram(Instagram):
    def __init__(self):
        super().__init__()
        print("NewInstagram")
        self.api = NewInstagramAPI()


class NewInstagramAPI(InstagramAPI):
    def __init__(self):
        super().__init__()
        print("NewInstagramAPI")
        self.url = "https://instagram.com/explore"

        # Work better with poor connections
        self.retry_rate = 10
        self.num_retries = 15
        if not NewInstagramAPI.proxies:
            self.readproxies()
    
    proxies = []
    @staticmethod
    def readproxies():
        print("fillproxies")
        with open("goodproxies.json","r") as file:   
            NewInstagramAPI.proxies = json.load(file)

    def api_call(self, edge, parameters, return_results=True):
        proxies = random.choice(self.proxies)
        print(str(proxies))
        req = self.get("%s/%s" % (self.url, edge),proxies=proxies, params=parameters,
                       headers={'Connection': 'close'})

        if not req:
            return None

        if return_results:
            try:
                return req.json()
            except json.JSONDecodeError:
                print("API response was not valid JSON")
                print(req.text)
                sleep(10)
                return self.api_call(edge, parameters, return_results=return_results)


api = NewInstagram()

# Get 1000 posts from #selfie
for post in api.hashtag("bubblecap", 1000):
    print(post)

