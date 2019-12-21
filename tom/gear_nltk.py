import redisAI
import nltk
from nltk.corpus import wordnet

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

def remove_stopwords(text,stopword):
    text = [word for word in text if word not in stopword]
    return text

def processText(x):
    ''' clean text for processing '''
    print(str(x))
    stopword = nltk.corpus.stopwords.words('english')
    wn = nltk.WordNetLemmatizer()
    caption, tags = extract_hashtags(x['caption'])
    caption, mentions = extract_mentions(caption)
    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stopword)
    caption = ' '.join(caption)
    print(str(caption))
    return caption

def runModel(x):
    ''' run onnx model '''
    modelRunner = redisAI.createModelRunner('auction:model')
    redisAI.modelRunnerAddInput(modelRunner, )
    
def storResults(x):
    ''' store to output stream '''

def printStuff(x):
    print(str(x))

gb = GearsBuilder('StreamReader')
gb.map(printStuff)
# gb.map(processText)
gb.register('post:')
# gb.map(processText)
# gb.map(runModel)
# gb.map(storResults)
# gb.register('camera:0')