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
# TODO: PERF METRICS / INSTRUMENTATION

def pre_proc_text(x):

    def extract_hashtags(caption, postId):
        regexp = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
        # tags = []

        def repl(m):
            # TODO: Done: INSERT TAGS INTO REDIS LIST HERE
            # DELETE THE LIST LATER IF NOT REQUIRED
            # TODO: GEARS PIPELINE
            # TODO: CHANGE SET TO PDD? 
            tag = m.group(0)[1:]
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
    
    postId = x['postId']
    rootTag = x['rootTag']
    caption = x['caption']
    # TODO: BREAK ON DUPE
    execute('cmsIncrBy', 'sketch:allposts', postId, '1')
    exists = execute('exists', f'sketch:all:{rootTag}')
    if exists:
        execute('cmsIncrBy', f'sketch:all:{rootTag}', postId, '1')
    # rootTag = bubblecap
    # tag = 
    print('ONE')
    caption = extract_hashtags(caption, postId)
    
    caption_text = extract_mentions(caption)
    caption = caption_text

    caption = remove_punct(caption)
    caption = tokenize(caption)

    caption = remove_stopwords(caption, stopwords)
    caption = ' '.join(caption)
    caption = caption.rstrip()
    print('TWO')
    # TODO: DONE: DON'T RETURN TAGS. THEY'RE NOW IN REDIS LIST
    
    return x, caption, caption_text

def runModel(x):
    # start = time.perf_counter()
    caption = x[1]
    sample = vectorizer.transform([caption]).toarray()
    ba = np.asarray(sample, dtype=np.float32)
    print(caption)
    res = execute('AI.TENSORSET', 'auction:tensor', 'FLOAT',
        '1', '80', 'BLOB', ba.tobytes())
    print(f'res1: {res}')

    res2 = execute('AI.MODELRUN', 'auction:model',
        'INPUTS', 'auction:tensor',
        'OUTPUTS', 'out_label', 'out_probs')
    print(f'res2: {res2}')

    out = execute('AI.TENSORGET', 'out_label', 'VALUES')

    print(f'out: {out}')
    print(f'out2 {out[2]}')
    print(f'pred {out[2][0]}')
    # end = time.perf_counter()
    # print(f'NEW: {end - start}')
    # print(out)
    return out[2][0]

def storeResults(x):
    ''' store to output stream '''
    # start = time.perf_counter() # time 0.0001
    # print(len(x[2]))
    # TODO: PIPELINE
    # TODO: rootTag should be the hashtag that initialised the search.
    # subtags should be saved to parent related and not stored unless unique
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
    
    # end1 = time.perf_counter()
    # print(f'Timer1: {end1 - start}')

    # TODO: REFACTOR AND MOVE THIS OUT OF THE GEAR
    # AND IN TO THE CACHE HANDLER

    # TODO: DONE: CACHE MAKER DECIDES IF ANYTHING TO CACHE
    # TODO: GEAR SHOULD NOTIFY CACHE MAKER
    # TODO: ADD STREAMKEY TO LIST TO TRIGGER FURTHER PROCESSING
    # TODO: USE A PDS TO TIME/BENCHMARK...
    execute('topk.add', 'topk:10tags', rootTag)
    execute('topk.add', 'topk:100tags', rootTag)
    # TODO: replace zset with HLL/CMS ordered 
    execute('zincrby', 'topzhashtags:', rootTag, 1)
    # TODO: Sketch must be created first
    if not execute('exists', f'sketch:out:{rootTag}'):
        print(f'creating new sketch: {rootTag}')
        execute('cmsInitByDim',
            f'sketch:out:{rootTag}', 2000, 10)
    execute('cmsIncrBy', f'sketch:out:{rootTag}', postId, 1)
    execute('cmsIncrBy', 'sketch:predicted', postId, 1)
    
# TODO: AVOID STORING SAME POST TWICE

gb = GearsBuilder('StreamReader')
gb.map(pre_proc_text)
gb.filter(runModel)
gb.foreach(storeResults)
gb.register('post:')

# TODO: look for words in sets 
# E.G. "FOR SALE" "SOLD" "BUY NOW" "BID" "DIBS"
# IF IT EXISTS STORE THEM
