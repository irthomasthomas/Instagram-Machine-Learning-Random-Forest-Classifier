import redisAI as rai
import redis
from stop_words import get_stop_words
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

vectorizer = pickle.load(open("/etc/tommy/vectorizer.pickle", "rb"))
conn = redis.Redis(host="localhost", port=6379, db=0)


def pre_proc_text(x):

    def extract_hashtags(caption):
        regexp = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
        tags = []

        def repl(m):
            tags.append(m.group(0))
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
        return caption, tags

    def remove_punct(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def tokenize(text):
        text = re.split('\W+', text)
        return text

    def remove_stopwords(text, stopword):
        text = [word for word in text if word not in stopword]
        return text

    caption = x['caption']
    stopwords = get_stop_words('english')
    caption, tags = extract_hashtags(caption)
    caption, mentions = extract_mentions(caption)
    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stopwords)
    caption = ' '.join(caption)
    caption = caption.rstrip()
    # print(caption)
    
    return x['streamId'], caption


def runModel(x):
    ''' run onnx model '''
    print(str(x))
    print('model start')
    input_name = 'float_input'
    label_name = 'output_label'
    vectorizer = pickle.load(open("/etc/tommy/vectorizer.pickle", "rb"))

    # tensor = redisAI.createTensorFromValues(
    #     'FLOAT', 1, 80)
    print('vectorizer loaded')
    modelRunner = rai.createModelRunner('auction:model')
    print('modelRunner set')
    # text = x['caption']
    text = x
    print(text)
    X = vectorizer.transform([text]).toarray()
    print("vectorized")
    
    X_ba = bytearray(X.tobytes())
    # tensor = rai.createTensorFromValues('INT32', [1, 80], X_ba)
    tensor = rai.createTensorFromBlob('FLOAT', [1, 80], X_ba)
    print("tensor")
    rai.modelRunnerAddInput(modelRunner, input_name, tensor)
    print("add input")
    rai.modelRunnerAddOutput(modelRunner, label_name)
    # rai.modelRunnerAddOutput(modelRunner, prob_label)
    print("add output")

    model_replies = rai.modelRunnerRun(modelRunner) #error expected 2 outputs got one
    print("model replies")
    print(str(model_replies))
    model_output = model_replies[0]
    print("model output")
    print(str(model_output))


def runModel2(x):
    ref, caption = x[0], x[1]
    sample = vectorizer.transform(
        [caption]).toarray()
    ba = np.asarray(sample, dtype=np.float32)
    print("ba")
    conn.execute_command(
        'AI.TENSORSET', 'auction:tensor', 'FLOAT',
        '1', '80', 'BLOB', ba.tobytes())
    print("tensor set")
    conn.execute_command(
        'AI.MODELRUN', 'auction:model', 'INPUTS', 'auction:tensor', 'OUTPUTS', 'out_label', 'out_probs')
    print("model run")
    out = conn.execute_command(
        'AI.TENSORGET', 'out_label', 'VALUES')
    print("tensorget")
    print(out[2])
    
def storeResults(x):
    ''' store to output stream '''

def printStuff(x):
    print(str(x))

# .map(lambda r : str(r)) # transform a Record into a string Record
# .foreach(
#   lambda x: redisgears.execute_command(
#       'set', x['value'], x['key'])) 
# # will save value as key and key as value
# redisgears.executeCommand('xadd', 'cats', 'MAXLEN', '~', '1000', '*', 'image', 'data:image/jpeg;base64,' + base64.b64encode(x[1]).decode('utf8'))
# res_id = execute('XADD', 'camera:0:yolo', 'MAXLEN', '~', 1000, '*', 'ref', ref_id, 'boxes', boxes, 'people', people)
# "postId": id, "likes": likes, "comments":comments,"caption":caption, 
#         "typename":typename,"owner_id":owner_id,"shortcode":shortcode,"timestamp":timestamp,"scrape_date":scrape_date
gb = GearsBuilder('StreamReader')
gb.map(pre_proc_text)
gb.map(runModel2)
gb.register('post:')

# gb.map(printStuff)

# gb.map(storResults)


# gb.map(runModel)