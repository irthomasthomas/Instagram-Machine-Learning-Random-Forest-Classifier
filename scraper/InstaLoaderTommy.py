import json
import pickle
import random
import re
import time
from multiprocessing import Process
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import requests
from redis import Redis

from instaloader import Instaloader, InstaloaderContext, Post
from instaloader.exceptions import *
from instaloader.structures import PostComment, PostCommentAnswer, Profile
from redisbloom.client import Client
from rejson import Client as rjclient
from rejson import Path

rj = rjclient(decode_responses=True)
rb = Client(decode_responses=True)
rdb = Redis(host='localhost', port=6379, db=0, decode_responses=True)


def post_dict(edge, hashtag, is_first_page, related=False):
    # TODO: PYCURL TO REDISJSON
    # TODO: REFACTORING
    # post['is_first_page'] = is_first_page
        # test = post['is_first_page']
        # print(f'test: is first page: {test}') # True
    try:
        postId = edge['node']['id']
    except:
        postId = ""
    try:
        likes = edge['node']['edge_liked_by']['count']
    except:
        print('empty')
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
    try:
        imgUrl = f'https://www.instagram.com/p/{shortcode}/media/?size=m'
    except:
        imgUrl = ""
    if is_first_page:
        is_first_page = 'True'
    else:
        is_first_page = 'False'
    if related:
        related_tag = related
    else:
        related_tag = 'False'

    return {"postId": postId,
        "rootTag": hashtag, "related_tag": related_tag,
        "is_first_page": is_first_page, "imgUrl": imgUrl,
        "likes": likes, "comments":comments,
        "caption":caption, "typename":typename,
        "owner_id":owner_id,"shortcode":shortcode,
        "timestamp":timestamp,"scrape_date":scrape_date}

# TODO: DUMP PAGE TO RedisJSON
def save_page_to_redis(page, hashtag, is_first_page, dump_page=False):
    # TODO[x] -Done: Use related stream for given tag
    # TODO -Done: gear.py roottag
    # print(f'save_to_redis:  tag: {hashtag} is_first: {is_first_page}')
    print('saving to redis')
    pipe = rdb.pipeline()
    i = 1

    for edge in page['edges']:
        post = post_dict(edge, hashtag, is_first_page)
        pipe.xadd("post:", post, maxlen=2000)
        if i > 5:
            pipe.execute() 
            i = 1
        else:
            i += 1 
    pipe.execute()

    if dump_page and is_first_page:
        key = f'json:page:{hashtag}'
        print(f'saving page to {key}')
        rj.jsonset(key, Path.rootPath(), page)
        print('done saving.')
        # extract_hashtags(page=page, rootTag=hashtag)
    
def save_archiver_to_redis(page, hashtag):
    # TODO: Done: Use related stream for given tag
    pipe = rdb.pipeline()
    i = 1
    for edge in page['edges']:
        post = post_dict(
            edge=edge, hashtag=hashtag,
            is_first_page=False)
        post['archiver'] = 'True'
        pipe.xadd("post:", post, maxlen=2000)
        if i > 5:
            pipe.execute()
            i = 1
        else:
            i += 1 
    pipe.execute()

def save_related_result(page, relatedTag, rootTag):
    pipe = rdb.pipeline()
    i = 1
    for edge in page['edges']:
        post = post_dict(
            edge=edge, hashtag=rootTag,
            is_first_page=False, related=relatedTag)

        pipe.xadd("post:", post, maxlen=2000)
        if i > 5:
            pipe.execute()
            i = 1
        else:
            i += 1 
    pipe.execute()

def default_user_agent() -> str:
    print(f'FUNC: default_user_agent')
    return 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' \
           '(KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36'

def get_end_cursor(hashtag):
    # TODO: Done: cursors to redis
    return True



class InstaloaderTommy(Instaloader):
    def __init__(self, mode="normal"):
        super().__init__('Instaloader')
        if mode == 'burst':
            self.context = InstaloaderContextTommy(max_connection_attempts=1)
        else:
            self.context = InstaloaderContextTommy(max_connection_attempts=10)
        self.download_comments=True
        print("CLASS INIT: InstaloaderTommy Inint")


    def update_comments(self, filename: str, post: Post) -> None:

        def _postcommentanswer_asdict(comment):
            return {'id': comment.id,
                    'created_at': int(comment.created_at_utc.replace(tzinfo=timezone.utc).timestamp()),
                    'text': comment.text,
                    'owner': comment.owner._asdict(),
                    'likes_count': comment.likes_count}

        def _postcomment_asdict(comment):
            return {**_postcommentanswer_asdict(comment),
                    'answers': sorted([_postcommentanswer_asdict(answer) for answer in comment.answers],
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
                else:
                    unique_comments_list[-1]['likes_count'] = y.get('likes_count')
                    if combine_answers:
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


    def get_hashtag_posts(
            self, hashtag: str, archive_run=False,
            related_burst=False, dump_page=False,
            duplicate_check=True, resume=False,
            end_cursor=False, is_root_tag=False) -> Iterator[Post]:
        """Get Posts associated with a #hashtag."""
        # print("GET_HASHTAG_POSTS()")
        if related_burst:
            retries = 1
            timeout = 1
        
        has_next_page = True
        is_first_page = False
        if resume:
            end_cursor = rdb.get(f'page:cursor:{hashtag}')
            # end_cursor = get_end_cursor(hashtag)
        while has_next_page:
            if end_cursor:
                params = {'__a': 1, 'max_id': end_cursor}
                is_first_page = False
            else:
                params = {'__a': 1}
                is_first_page = True
            hashtag_data = self.context.get_json(
                f'explore/tags/{hashtag}/',
                params)['graphql']['hashtag']['edge_hashtag_to_media']       
            
            count = hashtag_data['count'] # TODO: USE PAGE COUNT TO DECIDE SCRAPE DEPTH
            # print(f'count: {count}')
            end_cursor = hashtag_data['page_info']['end_cursor']
            if end_cursor:
                rdb.set(f'page:cursor:{hashtag}', end_cursor)
            else:
                rdb.set(f'scrape:complete:{hashtag}', 'True')
            postId = hashtag_data['edges'][0]['node']['id']
            # if the postId exists for rootTag we stop
            # if it exists from other tag skip prediction
            # possible states...
                # New tag, 
                    # no rootTag
                    # rootTag = tag
                    # 1st request
                    # 1st page
                    # store pageid for resume
                        # req level 2 related tags
                        # stop after some number of scrapes
                        # 
                        # stop scraping based on feedback from redisAI
                        # use relevance score
                        # store frequency score
                        # found / searched 600 / 20 = 30
                        # 6000 / 7 = 857 
                        # 60 / 
                        # if freq_score > 600 stop scraping, mark irrelivant
                # level 2 req related tags
                    # rootTag = rootTag
                    # level2/related req = True
                    # only scrape 1 page and return
                    # if 1st page pred1's > 1 carry on, stop on 0
                    # do not add related tags to scrape queue
                # Archive scrape
                    # get pageid
                    # scrape resume
                    # scrape 10 pages
                    # store pageid
                    # exit
            # TODO: Are tags being looped twice in scraper and in gears?
            # TODO: Done: Get page 1, then get 1 page for each tag found
            # CHECK IF A CMS EXISTS FOR THIS HASHTAG
            # IF NOT CREATE ONE
            # TODO: Done: Don't make sketch if burst run

            tag_sketch_key = f'sketch:all:{hashtag}'
            if rdb.exists(tag_sketch_key):
                print(f'exists: {tag_sketch_key}')
                post_exists = rb.cmsQuery(tag_sketch_key, postId)[0]
                print(f'post_exists: {post_exists}')
            else:
                print(f'sketch NOT Exists {tag_sketch_key}')
                res = rb.cmsInitByDim(tag_sketch_key, 2000, 10) # 0.01% Error rate
                print(f'cms res: {res}')
                post_exists = False

            if related_burst:
                print('RELATED_BURST')
                save_related_result(
                    hashtag_data, hashtag, is_root_tag)
            elif archive_run:
                print('ARCHIVE_RUN')
                save_archiver_to_redis(
                    hashtag_data, hashtag)
            else:
                print('INITIAL_SCRAPE')
                save_page_to_redis(
                    page=hashtag_data, hashtag=hashtag,
                    is_first_page=is_first_page, dump_page=dump_page)
            # TODO: DONE:ARCHIVER.py: HANDLE DEEP SCRAPING

            
            if post_exists and not archive_run:
                print('POST EXISTS')
                yield 6000            
            else:
                yield len(hashtag_data['edges'])
                
            has_next_page = hashtag_data['page_info']['has_next_page']
            print(f'has_next_page: {has_next_page}')
            # TODO: Done: If page 1 and not in requests set, extract hashtags and submit to burst scrape 1 page
            print(f'related_burst: {related_burst}')
           

            # if related_burst or archive_run:
            #     return
            # else:
            #     exctract_tags(hashtag_data, hashtag)
                # p_extract = Process(
                #         target=exctract_tags,
                #         args=(hashtag_data, hashtag))
                # p_extract.start()
            
            
# TODO: WHAT is smallest time unit in python?
class InstaloaderContextTommy(InstaloaderContext):
    default_user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'

    PROXIES_LIST = []
    
    def __init__(
            self, sleep: bool = True,
            quiet: bool = False,
            user_agent: Optional[str] = None,
            max_connection_attempts: int = 3):
        print("CONTEXT INIT")

        proxy = self.change_proxy()
        self.proxies = {'https' : proxy}
        # print(self.proxies)

        self.user_agent = user_agent if user_agent is not None else default_user_agent()
        self._session = self.get_anonymous_session()
        # print(f'_session: {self._session}')
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


    def change_proxy(self):
        '''Delete proxy and get new proxy'''
        proxy = rdb.zpopmin('proxies:')[0]        
        print(f'pop proxy: {proxy[0]}:{proxy[1]}')
        return proxy[0]


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
        # is_iphone_query = host == 'i.instagram.com'
        sess = session if session else self._session
        proxy = self.change_proxy()
        proxies = {"https" : proxy}
        print(f'get_json ')

        # for attempt_no in range(30):
        #     print(f'attempt: {attempt_no}')
        try:
            conn_start = time.perf_counter()
            resp = sess.get(
                f'https://{host}/{path}',
                proxies=proxies,
                params=params,
                allow_redirects=False,
                timeout=3)

            while resp.is_redirect:
                print("redirect")
                redirect_url = resp.headers['location']
                if redirect_url.startswith(f'https://{host}/'):
                    resp = sess.get(redirect_url if redirect_url.endswith('/') else redirect_url + '/',
                                    proxies=proxies, params=params, allow_redirects=False, timeout=3)
                else:
                    break
            conn_end = time.perf_counter()
            conn_time = conn_end - conn_start
            # print(f'conn_time: {conn_time}')
            # print(f'ADD {proxy}:{conn_time}')
            # TODO: Done: SET PROXY SCORE REDIS
                # if resp.status_code == 400:
                #     print("error 400")
                # if resp.status_code == 404:
                #     print("error 404")
                #     # raise QueryReturnedNotFoundException("404 Not Found")
                # if resp.status_code == 429:
                #     print("TEST error 429: too many requests")                    
                #     proxy = self.change_proxy()
                #     proxies = {"https" : proxy}
                #     continue
            if resp.status_code != 200:
                print(f'status code: {str(resp.status_code)} proxy:{proxies}')
                proxy = self.change_proxy()
                proxies = {"https" : proxy}
            else:                
                rdb.zadd('proxies:', {proxy: conn_time})


            is_html_query = not is_graphql_query and not "__a" in params and host == "www.instagram.com"
            if is_html_query:
                print('is_html')
                match = re.search(r'window\._sharedData = (.*);</script>', resp.text)
                if match is None:
                    print('match is None')
                    proxy = self.change_proxy()
                    proxies = {"https" : proxy}
                    return self.get_json(
                    path=path, params=params, host=host,
                    session=sess, _attempt=_attempt + 1)
                return json.loads(match.group(1))
            else:
                resp_json = resp.json()
            if 'status' in resp_json and resp_json['status'] != "ok":
                print(f'error: attempt_no {_attempt}')
                print(f'_attempt: {_attempt} PROXY: {proxies}')
                proxy = self.change_proxy()
                proxies = {"https" : proxy}
                # continue
                return self.get_json(
                    path=path, params=params, host=host,
                    session=sess, _attempt=_attempt + 1)

            return resp_json
        
        except KeyboardInterrupt as err:
            error_string = f'JSON Query to {path}: {err}'
            self.error("[skipped by user]", repeat_at_end=False)
            raise ConnectionException(error_string) from err
        except Exception as e:
            print(f'EXCEPTION: {e}')
            error_string = "JSON Query to {}: {}".format(path, e)
            if _attempt == self.max_connection_attempts:
                raise ConnectionException(error_string) from err
            print(f'error: attemp: {_attempt}')
            proxy = self.change_proxy()
            proxies = {"https" : proxy}
            return self.get_json(
                    path=path, params=params, host=host,
                    session=sess, _attempt=_attempt + 1)
        # except QueryReturnedNotFoundException as err:
        #     raise err

        # except (ConnectionException, json.decoder.JSONDecodeError, requests.exceptions.RequestException) as err:           
        #     error_string = f'JSON Query to {path}: {err}'
        #     if _attempt == self.max_connection_attempts:
        #         print(f'_attempt {_attempt} attempt_no: {attempt_no} max_connection_attempts {self.max_connection_attempts}')
        #         proxies = self.change_proxy(self.proxy)
        #     self.error(error_string + " [retrying; skip with ^C]", repeat_at_end=False)
        #     try:
        #         if is_graphql_query and isinstance(err, TooManyRequestsException):
        #             self._ratecontrol_graphql_query(params['query_hash'], untracked_queries=True)
        #         # if is_iphone_query and isinstance(err, TooManyRequestsException):
        #             # self._ratecontrol_graphql_query('iphone', untracked_queries=True)
        #         return self.get_json(path=path, params=params, host=host, session=sess, _attempt=_attempt + 1)
            # except KeyboardInterrupt:
            #     self.error("[skipped by user]", repeat_at_end=False)
            #     raise ConnectionException(error_string) from err


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
                                     owner=Profile(self._context, node['owner']),
                                     likes_count=node['edge_liked_by']['count'])


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
            pic_json = self._context.get_json(f'p/{self.shortcode}/', params={})
            self._full_metadata_dict = pic_json['entry_data']['PostPage'][0]['graphql']['shortcode_media']
            self._rhx_gis_str = pic_json.get('rhx_gis')
            if self.shortcode != self._full_metadata_dict['shortcode']:
                self._node.update(self._full_metadata_dict)
                raise PostChangedException
