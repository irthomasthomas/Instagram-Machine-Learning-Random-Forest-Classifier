from instaloader import Instaloader
from instaloader import InstaloaderContext, Post
from instaloader.structures import PostComment, PostCommentAnswer, Profile
from instaloader.exceptions import *

from typing import Any, Callable, Dict, Tuple, Iterator, List, Optional, Union
import requests
import json
import random
import redis
import time
import re

def post_dict(edge):
    # TODO: Handle empty ID

        # try:
        #     thumbnail_src = edge['node']['thumbnail_src']
        #     print(f'thumb: {thumbnail_src}')
        # except:
        #     print('no thumbnail_src')
        #     thumbnail_src = ""
        try:
            imgurl = edge['node']['thumbnail_resources'][2]['src']
        except:
            print('ERROR: no img340 at loc: edge[node][thumbnail_resources][2][src]')
            imgurl = ""
        try:
            id = edge['node']['id']
        except:
            id = ""
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
        return {"postId": id, "imgurl": imgurl, "likes": likes, "comments":comments,"caption":caption, 
        "typename":typename,"owner_id":owner_id,"shortcode":shortcode,"timestamp":timestamp,"scrape_date":scrape_date}
  

def save_post_to_redis(page):
    r = redis.Redis(host='localhost', port=6379, db=0)  
    pipe = r.pipeline()
    for edge in page['edges']:
        post = post_dict(edge)
        r.xadd("post:", post, maxlen=None)
    pipe.execute()

def default_user_agent() -> str:
    return 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' \
           '(KHTML, like Gecko) Chrome/51.0.2704.79 Safari/537.36'


def get_end_cursor(hashtag):
    # TODO: cursors to redis
    return True


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
        has_next_page = True
        if resume:
            end_cursor = False
            # end_cursor = get_end_cursor(hashtag)
        while has_next_page:
            if end_cursor:
                params = {'__a': 1, 'max_id': end_cursor}
            else:
                params = {'__a': 1}
            hashtag_data = self.context.get_json(
                f'explore/tags/{hashtag}/',
                params)['graphql']['hashtag']['edge_hashtag_to_media']       
            count = hashtag_data['count']     
            end_cursor = hashtag_data['page_info']['end_cursor']
            # TODO: end_cursor key epiry
            save_post_to_redis(hashtag_data)
            yield len(hashtag_data['edges'])
            has_next_page = hashtag_data['page_info']['has_next_page']

class InstaloaderContextTommy(InstaloaderContext):
    default_user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'

    proxies = []
    @staticmethod
    def fillproxies():
        with open("goodproxies.json", "r") as file:
            InstaloaderContextTommy.proxies = json.load(file)

    def __init__(self, sleep: bool = True, quiet: bool = False, user_agent: Optional[str] = None,
                 max_connection_attempts: int = 3):
        # _session = self._session

        # if not InstaloaderContextTommy.proxies:
        #     InstaloaderContextTommy.fillproxies()
        # self.user_agent = user_agent if user_agent is not None
        # else default_user_agent()
        print('INIT')
        if user_agent is not None:
            self.user_agent = user_agent
        else:
            self.user_agent = default_user_agent()

        # if not InstaloaderContextTommy._session:
        InstaloaderContextTommy._session = self.get_anonymous_session()

        # self._session = self.get_anonymous_session()
        self.username = None
        self.sleep = sleep
        self.quiet = quiet
        self.max_connection_attempts = max_connection_attempts
        self._graphql_page_length = 100
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
        # sess = session if session else self._session
        while True:
            try:
                proxy = random.choice(proxiesList)
                proxy = proxy[0] + ":" + proxy[1]
                proxies = {
                        "http" : proxy,
                        "https" : proxy
                        }
                break
            except:
                continue
        return proxies

    def get_json(
            self,
            path: str,
            params: Dict[str, Any],
            host: str = 'www.instagram.com',
            session: Optional[requests.Session] = None, 
            _attempt=1
            ) -> Dict[str, Any]:
       
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
        print('HELLO')
        is_graphql_query = 'query_hash' in params and 'graphql/query' in path
        is_iphone_query = host == 'i.instagram.com'
        sess = session if session else self._session
        proxies = {
            "https": '127.0.0.1:8888'
            }
        try:
            for attempt_no in range(100):
                resp = sess.get(
                    f'https://{host}/{path}',
                    proxies=proxies,
                    params=params,
                    allow_redirects=False,
                    timeout=3
                    )
                while resp.is_redirect:
                    redirect_url = resp.headers['location']
                    if redirect_url.startswith(f'https://{host}/'):
                        resp = sess.get(
                            redirect_url if redirect_url.endswith('/') 
                            else redirect_url + '/',
                            proxies=proxies,
                            params=params,
                            allow_redirects=False,
                            timeout=2
                            )
                    else:
                        break
                # if resp.status_code == 400:
                    # print("error 400")
                # if resp.status_code == 404:
                #     print("error 404")
                #     raise QueryReturnedNotFoundException("404 Not Found")

                # if resp.status_code == 429:
                #     print("TEST error 429: too many requests")
                if resp.status_code != 200:
                    print("GET_JSON ERROR: RESPONSE CODE:" + str(resp.status_code))
                    continue
                # is_html_query = not is_graphql_query and not "__a" in params and host == "www.instagram.com"
                # if is_html_query:
                #     match = re.search(r'window\._sharedData = (.*);</script>', resp.text)
                #     if match is None:
                #         continue
                #     return json.loads(match.group(1))
                else:
                    resp_json = resp.json()
                if 'status' in resp_json and resp_json['status'] != "ok":
                    continue
                    if 'message' in resp_json:
                        continue
                    else:
                        continue
                return resp_json
                    
        except QueryReturnedNotFoundException as err:
            raise err

        except (ConnectionException,
        json.decoder.JSONDecodeError,
        requests.exceptions.RequestException) as err:           
            error_string = f'JSON Query to {path}: {err}'
            if _attempt == self.max_connection_attempts:
                print(f'_attempt {_attempt} max_connection_attempts {self.max_connection_attempts}')
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
            pic_json = self._context.get_json(f'p/{self.shortcode}/', params={})
            self._full_metadata_dict = pic_json['entry_data']['PostPage'][0]['graphql']['shortcode_media']
            self._rhx_gis_str = pic_json.get('rhx_gis')
            if self.shortcode != self._full_metadata_dict['shortcode']:
                self._node.update(self._full_metadata_dict)
                raise PostChangedException
