from multiprocessing import Process, Queue, Manager, Pool
import multiprocessing as mp
import random
import sys
import os
import json
import time
import string
from multiprocessing.dummy import Pool as ThreadPool 
from instaloader import InstaloaderContext, Post
from typing import Any, Callable, Dict, Tuple, Iterator, List, Optional, Union
from instaloader.structures import PostComment, PostCommentAnswer, Profile

result = []


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
    

def load_json_from_fileobj(file):
    global result
    jfile = json.loads(file)
    result.append(jfile)


def fill_file_list(file, files):
    with open(file, 'r') as fp:
        files.append(fp.read())


def single_read():
    start = time.time()
    pool = Pool(2)
    jobs = []
    for file in os.scandir(path):
        with open(file, 'r') as fp:
            f = fp.read()
        p = Process(target=load_json_from_fileobj, args=(f,))
        jobs.append(p)
        p.start()
        p.join()
    print("single: end jobs[] len: " + str(len(jobs)))
    end = time.time()
    print(end - start)


def load_json_page(file):
    with open(file, 'r') as fp:
        f = fp.read()
        return json.loads(f)
        # posts.append(json.loads(f))


def load_json_posts_file_dir(path, posts):
        for file in os.scandir(path):
            with open(file, 'r') as fp:
                f = fp.read()
                posts.append(json.loads(f))


def single_thread_test(path, posts):
    start = time.time()
    for dir in os.scandir(path):
        load_json_posts_file_dir(dir, posts)
    print("single: end posts len: " + str(len(posts)))
    
    # end = time.time()
    print(time.time() - start)


def multi_threads_test(path):
    start = time.time()
    with Manager() as manager:
        posts = manager.list()
        # queue = Queue()
        processes = []
        for dir in os.scandir(path):
            for file in os.scandir(dir):
                processes.append(
                    Process(target=load_json_page, args=(file, posts)))

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print("multi: end posts len: " + str(len(posts)))
    print(time.time() - start)


def async_test(posts):
    start = time.time()
    with Manager() as manager:
        pool = Pool(processes=4)
        posts = manager.list()

        multiple_results = [
            pool.apply_async(
                load_json_posts_file_dir, args=[dir, posts]
                ) for dir in os.scandir(path)]
        # print([res.get(timeout=1) for res in multiple_results])
        print(len(posts))
        print(len(shared_list))
    print(time.time() - start)

    
def remove_stopwords(text, stopword):
    text = [word for word in text if word not in stopword]
    return text


def extract_hashtags(caption):
    regexp = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
    tags = []

    def repl(m):
        tags.append(m.group(0))
        return ""

    caption = regexp.sub(repl, caption.lower())
    return caption, tags


def remove_punct(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def tokenize(text):
    text = re.split('\W+', text)
    return text


def pre_proc_text(caption):
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    caption = re.split('\W+', caption)
    caption, tags = extract_hashtags(post['caption'])
    caption, mentions = extract_mentions(caption)
    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stopwords)
    caption = ' '.join(caption)

    return caption


if __name__ == "__main__":
    scraper = InstaloaderTommy()

    if len(sys.argv) > 1:
        path = sys.argv[1]
    files = []
    single_thread_test(path, files)
    posts = []
    for f in files:
        for edge in f:
                posts.append(TPost(scraper.context, edge['node']))
    print(len(files))
    print(len(posts))
    # pool = ThreadPool(len(files))
    # posts = [x for x in pool.map(load_json_page, files) if x is not None]

    # print(len(posts))
    # with Manager() as manager:
    #     posts = manager.list()
    #     pool = manager.Pool()
    #     for i in files:
    #         pool.apply_async(load_json_page, args=(i, posts))
    #         # pool.apply_async(load_json_page, args=(i, posts), callback=log_result)
    #     pool.close()
    #     pool.join()
    #     print(str(len(posts)))

        # multi_threads_test(path)
        # async_test(path)
    # print("posts: " + len(posts))
    # single_read()
    # print("single: end result[] len: " + str(len(result)))
