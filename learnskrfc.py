import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from time import time
import pickle
import csv
import re
import string
import time

# from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
# import tensorflow as tf
from stop_words import get_stop_words
import unidecode

# TODO: IDEA FOR APP SPLIT SCREEN SEARCH: OPEN DDG AND G SIDE-BY-SIDE


def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


def remove_stopwords(text, stopword):
    text = [word for word in text if word not in stopword]
    return text


def pre_proc_text(caption):

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

    def remove_accents(text):
        unaccented_string = unidecode.unidecode(text)
        return unaccented_string

    def tokenize(text):
        text = re.split('\W+', text)
        return text

    caption, tags = extract_hashtags(caption)
    caption, mentions = extract_mentions(caption)
    caption = remove_punct(caption)
    caption = remove_accents(caption)
    caption = tokenize(caption)
    # return [post[0], post[1], caption]
    return caption


def train_sk_rfc(input_file, stop_words):
    df = pd.read_csv(input_file, header=1)
    # output_file = "output2.csv" # 55k rows/4744 tagged id| og_caption| mod_caption| target
    df.columns = ["id", "og_caption", "mod_caption", "target"]

    df, y = df.og_caption, df.target

    documents = []
    for caption in range(0, len(df)):
        document = str(df[caption])
        document = pre_proc_text(document)
        document = remove_stopwords(document, stop_words)
        document = ' '.join(document)
        document = document.rstrip()
        documents.append(document)

    vectorizer = TfidfVectorizer(
        max_features=80, stop_words=stop_words)
    # vectorizer user warning stop_words inconsistent with preprocessing
    X = vectorizer.fit_transform(documents).toarray()
    print("SHAPE")
    print(X.shape)

    pickle.dump(vectorizer, open("vectorizer2020.pickle", "wb"))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=0)

    from sklearn.ensemble import RandomForestClassifier as rfc
    
    train_results = []
    test_results = []

    classifier = rfc(
        max_features=45, n_jobs=-1, 
        bootstrap=False, n_estimators=32, 
        random_state=0, max_depth=32, min_samples_leaf=5)
    classifier.fit(X_train, y_train)

    train_pred = classifier.predict(X_train)

    y_pred = classifier.predict(X_test)

    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"accuracy score: {accuracy_score(y_test, y_pred)}")
    # print('Mean Absolute Error:', mean_squared_error(y_test, y_pred))
    # print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)))
    cross_validate(X_train, y_train, classifier)
    with open("text_classifier2020", "wb") as picklefile:
        pickle.dump(classifier, picklefile)
        
    return classifier


def save_sklearn_to_onnx(classifier, output_filename):
    # Convert to ONNX format
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    # initial_type = [('float_input', FloatTensorType([1, 80]))]
    initial_type = [('float_input', FloatTensorType([None, 80]))]
    onx = convert_sklearn(classifier, initial_types=initial_type)
    with open(output_filename, "wb") as f:
        f.write(onx.SerializeToString())


def load_model_onnx(model_file):
    import onnxruntime as rt
    sess = rt.InferenceSession(model_file)
    return sess


def predict_and_merge(input_file, output_file, sess):
    timer1 = time.time()
    print("PREDICT AND MERGE")
    df = pd.read_csv(input_file, header=1)
    df.columns = ["MOD_CAP", "TRAIN", "FULL_CAP"]
    df, y = df.FULL_CAP, df.TRAIN

    test_documents = []
    for caption in range(0, len(df)):
        document = str(df[caption])
        document = pre_proc_text(document)
        document = remove_stopwords(document, stop_words)
        document = ' '.join(document)
        document = document.rstrip()

        test_documents.append(document)

    vectorizer = pickle.load(open("vectorizer2020.pickle", "rb"))

    X = vectorizer.transform(test_documents).toarray()

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    # pred_onx = sess.run(
        # [label_name], {input_name: X.astype(np.float32)})[0]
    # stimeronnx = time.time()
    pred_onx = sess.run(
        None, {input_name: X.astype(np.float32)})[0]
    print("pred_onx")
    print(str(pred_onx))
    # etimeronnx = time.time() - stimeronnx
    # print(f"onnx pred time: {etimeronnx}")

    with open("text_classifier2020", "rb") as training_model:
        model = pickle.load(training_model)
    # stimerog = time.time()
    prediction = model.predict(X)
    # etimerog = time.time() - stimerog
    # print(f"og model time: {etimerog}")
    # timer2 = time.time()
    # print(f"time1: {timer2 - timer1}")

    print(confusion_matrix(y, prediction))
    print(classification_report(y, prediction))
    print(f"accuracy_score: {accuracy_score(y, prediction)}")
    # timer3 = time.time()

    pred_df = pd.DataFrame()

    pred_df['caption'] = df
    pred_df['target'] = y
    pred_df['prediction'] = prediction
    pred_df['onnx'] = pred_onx

    pred_df.to_csv(output_file, header=True)
    # timer4 = time.time()
    # print(f"timer2: {timer4 - timer3}")

def predict(text, onx):
    vectorizer = pickle.load(open("vectorizer2020.pickle", "rb"))

    input_name = onx.get_inputs()[0].name
    label_name = onx.get_outputs()[0].name

    X = vectorizer.transform(text).toarray()
    pred_onx = onx.run(
        [label_name], {input_name: X.astype(np.float32)})[0]
    print(pred_onx)

def cross_validate(X_train, y_train, clf):
    from sklearn.model_selection import cross_val_score
    print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))

# take command of your life. cultivate self discipline.
    # output_file = "output2.csv" # 55k rows/4744 tagged id| og_caption| mod_caption| target
    # output 55k id | og_caption | mod_caption 
        # output1 55k id	og_caption	mod_caption	target 2061 1s 21k 2s
        # output2 55k 4700 1s 27k 0s 
    # train2.csv # 10k / 4828 mod_caption | target 
    # jk A data scientist is travelling on a train at 100 mph 
    # input_file = "ml data - full - balanced.csv"  # 7000 rows cleaned and cat  
    # input_file = "mlOutput - testSet.csv" # 700 rows cleaned and categorised
    # input_file = "~/mldata/ml data - full - biased - testSet.csv" # 1000 rows cleaned and cat

time0 = time.time()
stop_words = get_stop_words('english')
new_stop_words = []
for word in stop_words:
    new_stop_words.append(pre_proc_text(word)[0])
stop_words = new_stop_words
new_stop_words = 0

time01 = time.time()
print(f"time01: {time01 - time0}") # 0.019

input_file = "~/mldata/training_captions.csv"
classifier = train_sk_rfc(input_file, stop_words)

time02 = time.time()
print(f"time02: {time02 - time01}") # 60s

save_sklearn_to_onnx(classifier, "testclf72020.onnx")
sess = load_model_onnx("testclf72020.onnx")


time03 = time.time()
print(f"time03: {time03 - time02}")

timestr = time.strftime("%Y%m%d-%H%M%S")
output_file = f"mlout{timestr}.csv"

time1 = time.time()
print(f"time1: {time1 - time02}")
# test_file = "ml data - full - biased - testSet1.csv" # 10 rows cleaned
test_file = "~/mldata/testSet.csv"  # 1000 rows : MOD_CAP | TRAIN/category FULL_CAP | 150 1s, 750 0s

time11 = time.time()
print(f"time11: {time11 - time1}")
predict_and_merge(test_file, output_file, sess)

time12 = time.time()
print(f"time0: {time12 - time11}")
# Testing Samples
sample = "signed and dated 🔥ends when I call it 💎💎Starts at $5 and $5 usd min increments ( no reserve ) 🔥🔥Free shipping in the US ($10 usd to Mexico/ Canada ) 🔥💎please tag the person you outbid 💎failure to pay within 24 hours of winning auction or erasing bids = block 💎 thanks for all the support. Good luck 🍀👍🏻 #rasetglass #glassart #glassauction  #glassofig #glass #glassofig #glassforsale #glassart #glassblowing #glass_of_ig #pendysofig #pendys"
sample2 = 'Custom hand burned Shogun display box!!!💮🏯🎏🎋⛩ NFS #woodburning #colorado #boulder #woodencass #pine #woodwork #wood #japeneses #scarab #shogun #satisfying #woodart #handcarved #japeneseglass #woodworking #woodcarving #workshop #bestofglass #love #pin #katakana #woodcase #case #displaycase #engraving #dremel #woodartist #headyart #japanesestyle #srg2019'
sample = pre_proc_text(sample)
sample = remove_stopwords(sample, stop_words)
sample = ' '.join(sample)
sample = sample.rstrip()
sample = [sample]
predict(sample, sess)

time2 = time.time()
print(f"time2: {time2 - time12}")
print(f"time total: {time2 - time0}")