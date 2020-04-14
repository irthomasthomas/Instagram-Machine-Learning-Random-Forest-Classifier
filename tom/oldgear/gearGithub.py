from stop_words import get_stop_words
import re
import string
# from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import time
from redis import Redis

conn = Redis()

vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
stopwords = get_stop_words('english')

def pre_proc_text(x):

    def extract_hashtags(caption):
        regexp = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
        tags = []

        def repl(m):
            tag = m.group(0)[1:]
            tags.append(tag)
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
 
    caption = x['caption']

    caption, tags = extract_hashtags(caption)
    
    caption_text = extract_mentions(caption)
    caption = caption_text

    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stopwords)

    caption = ' '.join(caption)
    caption = caption.rstrip()

    return x, caption, caption_text, tags


def is_first_page(x):
    ''' If this is the first page
        add tags to queue:burst
        or else carry on through gear '''
    if x[0]['is_first_page'] == 'False':
        return x
    else:
        rootTag = x[0]['rootTag']
        t = 1
        for tag in x[3]:
            if tag == rootTag:
                continue
            key = f'root:tag:{tag}'
            execute('SET', key, rootTag)
            added = execute('SADD', 'queue:burst', tag)
            if added:
                execute('LPUSH', 'list:burst', tag)
            if t > 3:
                return x
            else:
                t += 1
        return x


def is_too_short(x):
    ''' If caption length is too small
        it's not worth predicting. '''
    if len(x[2]) > 15:
        return True
    else:
        return False

def runModel2(x):
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
    # TODO: PIPELINE

    rootTag = x[0]['rootTag']
    postId = x[0]['postId']
    shortcode = x[0]['shortcode']
    streamKey = f'tags:out:{rootTag}'
    link = f'https://www.instagram.com/p/{shortcode}'

    execute('XADD', streamKey,
        'MAXLEN', '~', 2500, '*',
        'link', link, 'caption_text', x[2],
        'rootTag', rootTag,
        'related_tag', x[0]['related_tag'],
        'postId', postId,
        'imgUrl', x[0]['imgUrl'],
        'likes', x[0]['likes'],
        'comments_count', x[0]['comments'],
        'shortcode', x[0]['shortcode'])

    res = execute('zincrby', 'topz:products', 1, rootTag)
    if x[0]['relatedTag'] == 'False':
        for tag in x[3]:
            execute('SADD', f'post:tags:{postId}', tag)
            execute('SADD', f'tag:posts:{tag}', postId)
        # TODO:STORE INFO ON RELATED in case one is hot
        execute('topk.add', 'top10tags', rootTag)
        execute('topk.add', 'top100tags', rootTag)
        
        execute('cms.IncrBy', f'sketch:out:{rootTag}', postId, '1') 
    

gb = GearsBuilder('StreamReader')
gb.map(pre_proc_text)
gb.map(is_first_page)
gb.filter(is_too_short)
gb.filter(runModel2)
gb.foreach(storeResults)
gb.register('post:')