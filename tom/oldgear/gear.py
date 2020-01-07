import redisAI as rai
from redis import Redis
from stop_words import get_stop_words
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import time

vectorizer = pickle.load(open("/home/tommy/development/vectorizer.pickle", "rb"))
conn = Redis(host="localhost", port=6379, db=0)
stopwords = get_stop_words('english')


def pre_proc_text(x):

    def extract_hashtags(caption, postId):
        regexp = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
        # tags = []

        def repl(m):
            tag = m.group(0)[1:]
            # tags.append(t)
            # TODO: INSERT TAGS INTO REDIS LIST HERE
            # DELETE THE LIST LATER IF NOT REQUIRED
            # TODO: GEARS PIPELINE
            execute('SADD', f'post:tags:{postId}', tag)
            execute('SADD', f'tag:posts:{tag}', postId)
            return ""

        caption = regexp.sub(repl, caption.lower())
        return caption

    def extract_mentions(caption):
        regexp = re.compile(r"(?:@)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
        tags = []

        def repl(m):
            tags.append(m.group(0))
            return ""

        caption = regexp.sub(repl, caption.lower())
        return caption

    def remove_punct(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def tokenize(text):
        # TODO: Anomalous backslash in string
        text = re.split('\W+', text)
        return text

    def remove_stopwords(text, stopword):
        text = [word for word in text if word not in stopword]
        return text
    
    # TODO: BREAK ON DUPE
    postId = x['postId']
    rootTag = x['rootTag']
    caption = x['caption']

    # execute('cmsIncrBy', 'sketch:allposts', postId, '1')
    # execute('cmsIncrBy', f'sketch:all:{rootTag}', postId, '1')


    caption = extract_hashtags(caption, postId)

    
    caption_text = extract_mentions(caption)
    caption = caption_text


    caption = remove_punct(caption)
    caption = tokenize(caption)


    caption = remove_stopwords(caption, stopwords)
    caption = ' '.join(caption)
    caption = caption.rstrip()

    # print(f'PRE PROCESSOR TIME: {end - start}')
    # print(f't1: {end2 - start} t2: {end3 - end2} t3: {end4 - end3}')

    # print(f'cap len: {len(caption)}') #not len slow 
    # print(f'tags len: {len(tags)}')
    # print(caption)
    # TODO: DON'T RETURN TAGS. THEY'RE NOW IN REDIS LIST
    
    return x, caption, caption_text

def runModel(x):
    # start = time.perf_counter()
    caption = x[1]
    print(caption)
    sample = vectorizer.transform([caption]).toarray()
    ba = np.asarray(sample, dtype=np.float32)
    execute(
        'AI.TENSORSET', 'auction:tensor', 'FLOAT',
        '1', '80', 'BLOB', ba.tobytes())
    execute(
        'AI.MODELRUN', 'auction:model',
        'INPUTS', 'auction:tensor',
        'OUTPUTS', 'out_label', 'out_probs')
    print(1)
    out = execute(
        'AI.TENSORGET', 'out_label', 'VALUES')
    # end = time.perf_counter()
    # print(f'NEW: {end - start}')
    print(out)

    return out[2][0]

def storeResults(x):
    ''' store to output stream '''
    # start = time.perf_counter() # time 0.0001
    # print(len(x[2]))
    # TODO: PIPELINE
    rootTag = x[0]['rootTag']
    postId = x[0]['postId']
    shortcode = x[0]['shortcode']
    streamKey = 'tags:out:' + rootTag
    link = f'https://www.instagram.com/p/{shortcode}'

    # execute('SADD', 'trackedTags', rootTag)
    execute('XADD', streamKey,
        'MAXLEN', '~', 1000, '*',
        'postId', x[0]['postId'],
        'imgUrl', x[0]['imgUrl'],
        'link', link,
        'rootTag', rootTag,
        'likes', x[0]['likes'],
        'comments_count', x[0]['comments'],
        'caption_text', x[3],
        'typename', x[0]['typename'],
        'owner_id', x[0]['owner_id'],
        'ig_timestamp', x[0]['timestamp'],
        'retrieved_date', x[0]['scrape_date'])
    
    # TODO: IF STREAM LEN > 10 SET CACHE FLAG
    # end1 = time.perf_counter()
    # print(f'Timer1: {end1 - start}')

    # TODO: REFACTOR AND MOVE THIS OUT OF THE GEAR
    # AND IN TO THE CACHE HANDLER

    # TODO: GEAR SHOULD NOTIFY CACHE MAKER
    # TODO: CACHE MAKER DECIDES IF ANYTHING TO CACHE
    execute('topk.add', 'top10tags', rootTag)
    execute('topk.add', 'top100tags', rootTag)
    execute('zincrby', 'topzhashtags:', rootTag, 1)
    execute('cmsIncrBy', f'sketch:out:{rootTag}', postId, '1')    
    # for tag in x[2]:
    #     execute('SADD', f'post:tags:{postId}', tag)
    #     execute('SADD', f'tag:posts:{tag}', postId)
    # Add The stream key to a stream
    # block read 1st stream
    # get stream key
    # end2 = time.perf_counter()
    # print(f'timer2: {end2 - end1}')

def printx(x):
    print(f'X: {x}')


gb = GearsBuilder('StreamReader')
gb.map(pre_proc_text)
gb.filter(runModel)
gb.foreach(storeResults)
gb.register('post:')

# TODO: look for words in sets 
# E.G. "FOR SALE" "SOLD" "BUY NOW" "BID" "DIBS"
# IF IT EXISTS STORE THEM