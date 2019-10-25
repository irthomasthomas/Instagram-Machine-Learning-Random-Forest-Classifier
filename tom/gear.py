import redisAI
# from nltk import WordNetLemmatizer
# from nltk.corpus import wordnet
# from nltk.corpus import stopwords
from stop_words import get_stop_words
import re
import string

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

    print(str(x['post_id']))
    # print(str(x['caption']))
    # stopword = stopwords.words('english')
    stop_words = get_stop_words('english')

    # wn = WordNetLemmatizer()
    caption, tags = extract_hashtags(x['caption'])
    caption, mentions = extract_mentions(caption)
    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stop_words)
    caption = ' '.join(caption)
    print(str(caption))
    return caption

def runModel(x):
    ''' run onnx model '''

    input_name = onx.get_inputs()[0].name # 'float_input'
    label_name = onx.get_outputs()[0].name # 'output_label'

    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([1, 80]))]
    onx = convert_sklearn(classifier, initial_types=initial_type)
    with open(output_filename, "wb") as f:
        f.write(onx.SerializeToString())

    tensor = redisAI.createTensorFromValues(
        'FLOAT', [1, 80], [80,80])
    modelRunner = redisAI.createModelRunner('auction:model')
    redisAI.modelRunnerAddInput(
        modelRunner, 'float_input', tensor)
    redisAI.modelRunnerAddOutput(modelRunner, 'output_label')
    model_replies = redisAI.modelRunnerRun(modelRunner)
    model_output = model_replies[0]    
    print(str(model_output))


    
def storResults(x):
    ''' store to output stream '''

def printStuff(x):
    print(str(x))

gb = GearsBuilder('StreamReader')
# gb.map(processText)
# gb.map(printStuff)
gb.map(processText)
gb.map(runModel)
gb.register('post:')

# gb.map(runModel)
# gb.map(storResults)
# gb.register('camera:0')