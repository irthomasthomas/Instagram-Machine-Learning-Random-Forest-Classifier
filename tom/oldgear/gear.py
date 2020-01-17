from stop_words import get_stop_words
import re
import string
# from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import time
from redis import Redis
conn = Redis()

vectorizer = pickle.load(open("/home/tommy/development/vectorizer.pickle", "rb"))
stopwords = get_stop_words('english')

word_list = ['sold', 'forsale', 'sale',
        'auction', 'bid', 'dibs', 'bidding',
        'selling', 'sell', 'pay', 'paypal',
        '$', '£', '€', 'dm', 'purchase']

def pre_proc_text(x):

    def extract_hashtags(caption):
        regexp = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
        tags = []

        def repl(m):
            tag = m.group(0)[1:]
            tags.append(tag)
            # TODO: INSERT TAGS INTO REDIS LIST HERE
            # DELETE THE LIST LATER IF NOT REQUIRED
            # TODO: GEARS PIPELINE
            return ""

        caption = regexp.sub(repl, caption.lower())
        return caption, tags

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
    # TODO: GET WORDS IN LIST
    #############################################
    
    caption = x['caption']
    # print(f'{postId}:{rootTag}')
    # execute('cmsIncrBy', 'sketch:allposts', postId, '1')
    # execute('cmsIncrBy', f'sketch:all:{rootTag}', postId, '1')

    caption, tags = extract_hashtags(caption)
    
    caption_text = extract_mentions(caption)
    caption = caption_text

    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stopwords)

    # TODO: WORD_LIST
        # word_freq = []
        # print(word_list)
        # for w in word_list:
        #     word_freq.append(word_list.count(w))
        # result = list(zip(word_list, word_freq))
        # print(result)
    
    caption = ' '.join(caption)
    caption = caption.rstrip()

    # print(f'PRE PROCESSOR TIME: {end - start}')
        # print(f't1: {end2 - start} t2: {end3 - end2} t3: {end4 - end3}')

        # print(f'cap len: {len(caption)}') #not len slow 
        # print(f'tags len: {len(tags)}')
        # print(caption)
    return x, caption, caption_text, tags


def is_first_page(x):
    ''' If this is the first page
        add tags to queue:burst
        or else carry on through gear '''
    # TODO: MOVE THIS TO SCRAPER
    # TODO: Ignore if roottag = newtag
    if x[0]['is_first_page'] == 'False':
        return x
    else:
        rootTag = x[0]['rootTag']
        # print(f'rootTag: {rootTag}')
        # print('is FIRST_PAGE') # ok
        t = 1
        for tag in x[3]:
            if tag == rootTag:
                continue
            key = f'root:tag:{tag}'
            # print(key) seems to work.
            execute('SET', key, rootTag)
            added = execute('SADD', 'queue:burst', tag)
            if added:
                execute('LPUSH', 'list:burst', tag)
            if t > 3:
                return x
            else:
                t += 1
        # print('1st page check')
        return x


def is_too_short(x):
    ''' If caption length is too small
        it's not worth predicting. '''
    if len(x[2]) > 15:
        return True
    else:
        return False


def runModel(x):
    # start = time.perf_counter()
    caption = x[1]
    sample = vectorizer.transform([caption]).toarray()
    ba = np.asarray(sample, dtype=np.float32)
    execute(
        'AI.TENSORSET', 'auction:tensor', 'FLOAT',
        '1', '80', 'BLOB', ba.tobytes())
    execute(
        'AI.MODELRUN', 'auction:model',
        'INPUTS', 'auction:tensor',
        'OUTPUTS', 'out_label', 'out_probs')
    out = execute(
        'AI.TENSORGET', 'out_label', 'VALUES')
    # end = time.perf_counter()
    # print(f'NEW: {end - start}')
    # print(out[2][0])
    # print(caption)
    # if out[2][0] != 0:
    rootTag = x[0]['rootTag']
    key = f'tag:out1:{rootTag}'
    print(key)
    execute('XADD', key, 'MAXLEN', '~', 2500, '*', 'res', out[2][0])
    print(out)
    
    return out[2][0]


def runModel2(x):
    # print(f'runModel: {x}')
    caption = x[1]
    sample = vectorizer.transform([caption]).toarray()
    ba = np.asarray(sample, dtype=np.float32)
    conn.execute_command(
        'AI.TENSORSET', 'auction:tensor', 'FLOAT',
        '1', '80', 'BLOB', ba.tobytes())
    conn.execute_command(
        'AI.MODELRUN', 'auction:model',
        'INPUTS', 'auction:tensor',
        'OUTPUTS', 'out_label', 'out_probs')
    out = conn.execute_command(
        'AI.TENSORGET', 'out_label', 'VALUES')
    return out[2][0]


def storeResults(x):
    ''' store to output stream '''
    # start = time.perf_counter() # time 0.0001
    # print(len(x[2]))
    # TODO: PIPELINE

    rootTag = x[0]['rootTag']
    postId = x[0]['postId']
    shortcode = x[0]['shortcode']
    streamKey = f'tags:out:{rootTag}'
    link = f'https://www.instagram.com/p/{shortcode}'


    # execute('SADD', 'trackedTags', rootTag)
    # TODO: Need to identify related
    execute('XADD', streamKey,
        'MAXLEN', '~', 2500, '*',
        'link', link, 'caption_text', x[2],
        'rootTag', rootTag,
        'related_tag', x[0]['related_tag'],
        'postId', postId,
        'imgUrl', x[0]['imgUrl'],
        'likes', x[0]['likes'],
        'comments_count', x[0]['comments'])
        # 'typename', x[0]['typename'],
        # 'owner_id', x[0]['owner_id'],
        # 'ig_timestamp', x[0]['timestamp'],
        # 'retrieved_date', x[0]['scrape_date'])
    
    # TODO: Done:Cache maker IF STREAM LEN > 10 SET CACHE FLAG
    # end1 = time.perf_counter()
    # print(f'Timer1: {end1 - start}')

    # TODO: DONE REFACTOR AND MOVE THIS OUT OF THE GEAR
    # AND IN TO THE CACHE HANDLER

    # TODO: DONE: CACHE MAKER CONTAINS CACHING LOGIC 
    
    res = execute('zincrby', 'topz:products', 1, rootTag)
    print(f'zincry response: {res}')
    print(x[0]['relatedTag'])
    if x[0]['relatedTag'] == 'False':
        for tag in x[3]:
            execute('SADD', f'post:tags:{postId}', tag)
            execute('SADD', f'tag:posts:{tag}', postId)
        # TODO:STORE INFO ON RELATED in case one is hot
        execute('topk.add', 'top10tags', rootTag)
        execute('topk.add', 'top100tags', rootTag)
        
        execute('cms.IncrBy', f'sketch:out:{rootTag}', postId, '1') 
    
    # try:
    #     relatedTag = x[0]['relatedTag']
    # except KeyError:

    # Add The stream key to a stream
    # block read 1st stream
    # get stream key
    # end2 = time.perf_counter()
    # print(f'timer2: {end2 - end1}')


gb = GearsBuilder('StreamReader')
gb.map(pre_proc_text)
gb.map(is_first_page)
gb.filter(is_too_short)
gb.filter(runModel2)
gb.foreach(storeResults)
gb.register('post:')

# TODO: look for words in sets 
# E.G. "FOR SALE" "SOLD" "BUY NOW" "BID" "DIBS"
# IF IT EXISTS STORE THEM