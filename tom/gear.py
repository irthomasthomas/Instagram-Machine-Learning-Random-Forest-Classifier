import redisAI as rai
# from nltk import WordNetLemmatizer
# from nltk.corpus import wordnet
# from nltk.corpus import stopwords
from stop_words import get_stop_words
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
# import pandas as pd


# from ml2rt import load_model

# init.py 
# input_stream_key = '{}:{}'.format(args.camera_prefix, args.camera_id)  # Input video stream key name
# initialized_key = '{}:initialized'.format(input_stream_key)
# conn = redis.Redis(host=url.hostname, port=url.port)
# Check if this Redis instance had already been initialized
    # initialized = conn.exists(initialized_key)
    # if initialized:
    #     print('Discovered evidence of a privious initialization - skipping.')
    #     exit(0)
# with open('models/tiny-yolo-voc.pb', 'rb') as f:
        # model = f.read()
        # res = conn.execute_command('AI.MODELSET', 'yolo:model', 'TF', args.device, 'INPUTS', 'input', 'OUTPUTS', 'output', model)
# print('Loading gear - ', end='')
# with open('gear.py', 'rb') as f:
#         gear = f.read()
#         res = conn.execute_command('RG.PYEXECUTE', gear)
# Lastly, set a key that indicates initialization has been performed
# print('Flag initialization as done - ', end='') 
# print(conn.set(initialized_key, 'most certainly.'))
       

def extract_hashtags(caption):

    def repl(m):
        tags.append(m.group(0))
        return ""

    regexp = re.compile(r"(?:#)(\w(?:(?:\w|(?:\.(?!\.))){0,28}(?:\w))?)")
    tags = []
   
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

def remove_stopwords(text,stopword):
    text = [word for word in text if word not in stopword]
    return text

def processText(x):
    ''' clean text for processing '''

    # print(str(x))
    print(str(x['post_id']))
    # print(str(x['caption']))
    # stopword = stopwords.words('english')
    stop_words = get_stop_words('english')
    # wn = WordNetLemmatizer()

    caption, tags = extract_hashtags(x['caption'])
    # print(str(x['caption']))
    caption, mentions = extract_mentions(caption)

    caption = remove_stopwords(caption, stop_words)
    print(caption)

    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = ' '.join(caption)
    print(str(caption))

    # execute('hmset', x['post_id'], 'caption', caption)
    msg = {
            'streamId': x['streamId'],
            'scrape_date': x['scrape_date'],
            'caption': caption,
            'post_id': x['post_id']
        }

    return msg


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
    print(caption)
    
    # return [post[0], post[1], caption]
    return caption


def runModel(x):
    ''' run onnx model '''
    # print(str(x))
    # print('model start')

    # input_name = onx.get_inputs()[0].name # 'float_input'
    # label_name = onx.get_outputs()[0].name # 'output_label'
    input_name = 'float_input'
    label_name = 'output_label'
    prob_label = 'output_probability'
    vectorizer = pickle.load(open("/etc/tommy/vectorizer.pickle", "rb"))

    # tensor = redisAI.createTensorFromValues(
    #     'FLOAT', 1, 80)
    # print('vectorizer loaded')
    # 'model:poisson1'
    modelRunner = rai.createModelRunner('auction:model')
    # print('created model runner')

    text = x['caption']
    X = vectorizer.transform([text]).toarray()
    print("vectorized")
    # X = X.astype(np.float32)
    
    X_ba = bytearray(X.tobytes())
    # tensor = rai.createTensorFromValues('INT32', [1, 80], X_ba)
    tensor = rai.createTensorFromBlob('FLOAT', [1, 80], X_ba)
    print("tensor")
    rai.modelRunnerAddInput(modelRunner, input_name, tensor)
    print("add input")
    rai.modelRunnerAddOutput(modelRunner, label_name)
    rai.modelRunnerAddOutput(modelRunner, prob_label)
    print("add output")

    model_replies = rai.modelRunnerRun(modelRunner) #error expected 2 outputs got one
    print("model replies")
    # model_output = model_replies[0]
    print("model output")
    # print(str(model_output))

    # pred_onx = onx.run(
    #     [label_name], {input_name: X.astype(np.float32)})[0]
    
    # rai.modelRunnerAddInput(modelRunner, 'input', X_ba)

    # name: Key on which tensor is saved
    # tensor: a Tensor object
    # shape: Shape of the tensor
    # dtype: redisai.DType object represents data type of the tensor. 
    # Required if input is a list/tuple
    # rai.tensorset("tensor_test", X, [1,80], dtype=rai.DType.float)

    # tensor = rai.createTensorFromValues('FLOAT', [1,80], X)
    # arr = np.array([1, 80])
    # tensor = rai.BlobTensor.from_numpy(arr)
    # rai.tensorset('x', tensor)
    print("set tensor")

    # print(str(X))
    # X.astype(np.float32)
    # tensor = rai.createTensorFromValues(
    #     'FLOAT', [1,80], X.astype(np.float32))
    # tensor = rai.createTensorFromBlob('FLOAT', [1, 80], X.astype(np.float32))


    # tensor = rai.Tensor.scalar(rai.DType.float, X.astype(np.float32))
    # tensor = rai.BlobTensor.from_numpy(X.astype(np.float32))

    # tensor = rai.BlobTensor.from_numpy(dummydata)
    # pred_onx = onx.run(
        # [label_name], {input_name: X.astype(np.float32)})[0]
        # from ml2rt import load_model
        # from cli import arguments
        # tensor = BlobTensor.from_numpy(np.ones((1, 13), dtype=np.float32))
        # model = load_model('../models/sklearn/boston_house_price_prediction/boston.onnx')
        # con.tensorset('tensor', tensor)
        # con.modelset('my_model', Backend.onnx, device, model)
        # con.modelrun('my_model', inputs=['tensor'], outputs=['out'])
        # out = con.tensorget('out', as_type=BlobTensor)
        # print(out.to_numpy())
        # # dummydata taken from sklearn.datasets.load_boston().data[0]
        # dummydata = [
        #     0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98]
        # tensor = rai.Tensor.scalar(rai.DType.float, *dummydata)
        # con.tensorset("input", tensor)
        # con.modelrun("sklearn_model", ["input"], ["output"])
        # outtensor = con.tensorget("output", as_type=rai.BlobTensor)

        # # dummydata = np.array(
        # #     [[-0.222222, 0.5, -0.762712, -0.833333]], dtype=np.float32)
        # # tensor = rai.BlobTensor.from_numpy(dummydata)
        # prime_tensor = rai.Tensor(rai.DType.int64, shape=(1,), value=5)

        # img = np.array(img_jpg).astype(np.float32)
        # img = np.expand_dims(img, axis=0)
        # img /= 256.0

        # tensor = rai.BlobTensor.from_numpy(img)
        # con.tensorset('in', tensor)
        # con.modelrun('yolo', 'in', 'out')
        # con.scriptrun('yolo-post', 'boxes_from_tf', inputs='out', outputs='boxes')
        # boxes = con.tensorget('boxes', as_type=rai.BlobTensor).to_numpy()

    # rai.modelRunnerAddInput(
    #     modelRunner, 'float_input', tensor)
    # print('model runner added inputs')
    
    # rai.modelRunnerAddOutput(modelRunner, 'output_label')
    # print('model runner added inputs')

    # model_replies = rai.modelRunnerRun(modelRunner)
    # model_output = model_replies[0]    
    # print(str(model_output))


    
def storeResults(x):
    ''' store to output stream '''

def printStuff(x):
    print(str(x))


gb = GearsBuilder('StreamReader')
# gb.map(processText)
# gb.map(printStuff)

gb.map(pre_proc_text)
# gb.map(printStuff)

# gb.map(runModel)
# gb.map(storResults)

gb.register('post:')

# gb.map(runModel)
# gb.register('camera:0')