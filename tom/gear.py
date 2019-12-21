import redisAI as rai
import redis
from stop_words import get_stop_words
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np

vectorizer = pickle.load(open("/home/tommy/development/vectorizer.pickle", "rb"))
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

    print(f'pre_proc_text: x {str(x)}')
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


# def runModel(x):
    #     ''' run onnx model '''
    #     print(str(x))
    #     print('model start')
    #     input_name = 'float_input'
    #     label_name = 'output_label'
    #     vectorizer = pickle.load(open("/etc/tommy/vectorizer.pickle", "rb"))

    #     # tensor = redisAI.createTensorFromValues(
    #     #     'FLOAT', 1, 80)
    #     modelRunner = rai.createModelRunner('auction:model')
    #     # text = x['caption']
    #     text = x
    #     X = vectorizer.transform([text]).toarray()
    #     print("vectorized")
        
    #     X_ba = bytearray(X.tobytes())
    #     # tensor = rai.createTensorFromValues('INT32', [1, 80], X_ba)
    #     tensor = rai.createTensorFromBlob('FLOAT', [1, 80], X_ba)
    #     rai.modelRunnerAddInput(modelRunner, input_name, tensor)
    #     rai.modelRunnerAddOutput(modelRunner, label_name)
    #     # rai.modelRunnerAddOutput(modelRunner, prob_label)

    #     model_replies = rai.modelRunnerRun(modelRunner) #error expected 2 outputs got one
    #     model_output = model_replies[0]
    #     print(str(model_output))


def runModel2(x):
    ref, caption = x[0], x[1]
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
    
    print(out[2])


def storeResults(x):
    ''' store to output stream '''
    execute('SADD', 'allposts', x['key'])
    return x


gb = GearsBuilder('StreamReader')
gb.map(pre_proc_text)
gb.map(runModel2)
# gb.map(runModel)
gb.register('post:')


# gb.map(storResults)


# gb.map(runModel)