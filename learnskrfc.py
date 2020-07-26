import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from time import time
import pickle
import csv
import re
import string

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


def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


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
    
    def remove_stopwords(text, stopword):
        text = [word for word in text if word not in stopword]
        return text

    stop_words = get_stop_words('english')
    # stop_words = stopwords.words('english')
    caption, tags = extract_hashtags(caption)
    caption, mentions = extract_mentions(caption)
    caption = remove_punct(caption)
    # caption = remove_accents(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stop_words)

    caption = ' '.join(caption)
    caption = caption.rstrip()
    
    # return [post[0], post[1], caption]
    return caption

def train_sk_rfc(input_file):

    df = pd.read_csv(input_file, header=None)
    df.columns = ["data", "target"]

    df, y = df.data, df.target

    documents = []
    for caption in range(0, len(df)):
        document = str(df[caption])
        # document = pre_proc_text(document)
        documents.append(document)
    # print(df)
    # Convert to numbers with bag of words
    

    start = time()
    stop_words = get_stop_words('english')
    
    vectorizer = TfidfVectorizer(
        max_features=80, stop_words=stop_words)
    print(str(time() - start))

    # vectorizer user warning stop_words inconsistent with preprocessing
    X = vectorizer.fit_transform(documents).toarray()

    print("SHAPE")
    print(X.shape)

    print(str(time() - start))

    pickle.dump(vectorizer, open("vectorizer2020.pickle", "wb"))
    print(str(time() - start))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

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

    print(str(time() - start))
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    # print('Mean Absolute Error:', mean_squared_error(y_test, y_pred))
    # print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)))
    # cross_validate(X_train, y_train, classifier)
    with open("text_classifier2020", "wb") as picklefile:
        pickle.dump(classifier, picklefile)
        
    return classifier


def save_sklearn_to_onnx(classifier, output_filename):
    # Convert to ONNX format 
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([1, 80]))]
    onx = convert_sklearn(classifier, initial_types=initial_type)
    with open(output_filename, "wb") as f:
        f.write(onx.SerializeToString())


def load_model_onnx(model_file):
    import onnxruntime as rt
    sess = rt.InferenceSession(model_file)
    return sess


def predict_and_merge(input_file, output_file, sess):
    start = time()

    df = pd.read_csv(input_file, header=None)
    df.columns = ["data", "target"]

    df, y = df.data, df.target

    test_documents = []
    for caption in range(0, len(df)):
        document = str(df[caption])
        test_documents.append(document)

    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))

    X = vectorizer.transform(test_documents).toarray()

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run(
        [label_name], {input_name: X.astype(np.float32)})[0]
    print(str(pred_onx))
    print(str(time() - start))

    with open("text_classifier", "rb") as training_model:
        model = pickle.load(training_model)

    # prediction = pd.DataFrame(pred_onx)
    prediction = model.predict(X)
    # prediction = pd.DataFrame(model.predict(X))

    print(confusion_matrix(y, prediction))
    print(classification_report(y, prediction))
    print(accuracy_score(y, prediction))

    pred_df = pd.DataFrame()

    pred_df['data'] = df
    pred_df['target'] = y
    pred_df['prediction'] = prediction
    pred_df['onnx'] = pred_onx

    pred_df.to_csv(output_file, header=True)
    print(str(time() - start))


def predict(text, onx):
    # vectorizer = pickle.load(open("/etc/tommy/vectorizer.pickle", "rb"))
    vectorizer = pickle.load(open("vectorizer2020.pickle", "rb"))
    # vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    # stop_words = get_stop_words('english')

    # vectorizer = TfidfVectorizer(
    #     max_features=80, stop_words=stopwords.words('english'))
    print(type(text))
    input_name = onx.get_inputs()[0].name
    label_name = onx.get_outputs()[0].name
    print(input_name)
    print(label_name)
    # input = [text,]
    X = vectorizer.transform(text).toarray()
   
    pred_onx = onx.run(
        [label_name], {input_name: X.astype(np.float32)})[0]
    print(pred_onx)
   
    
    # with open("text_classifier", "rb") as training_model:
    #     model = pickle.load(training_model)

    # prediction = model.predict(X)
    # print(str(prediction))

def cross_validate(X_train, y_train, clf):
    from sklearn.model_selection import cross_val_score
    # from sklearn.datasets import fetch_covtype
    # from sklearn import grid_search
    print(np.mean(cross_val_score(clf, X_train, y_train, cv=10)))


def train_models(input_file, test_file):
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC, LinearSVC, NuSVC
    from sklearn.metrics import accuracy_score, classification_report
    
    train = pd.read_csv(input_file, header=None)
    test = pd.read_csv(test_file, header=0)
    train.columns = ["data", "target"]
    print(train.describe())
    
    train, y = train.data, train.target

    documents = []
    for caption in range(0, len(train)):
        document = str(train[caption])
        documents.append(document)
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    
    X = vectorizer.fit_transform(documents).toarray()
    
    # features = train.iloc[:,0:1]
    # labels = ["data", "target"]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=0)

    x1, x2, y1, y2 = train_test_split(
        X, y, random_state=0, test_size=0.2
        )
    print(x1.shape)
    print(x2.shape)
    print(y1.shape)
    print(y2.shape)
    
    LSVC = LinearSVC()
    LSVC.fit(x1,y1)
    y2_LSVC_model = LSVC.predict(x2)
    print("LSVC Accuracy :", accuracy_score(y2, y2_LSVC_model))


# take command of your life. cultivate self discipline.

# input_file = "ml data - full - balanced.csv"
# input_file = "mlOutput - testSet.csv"
input_file = "~/mldata/ml data - full - biased - testSet.csv"
# stopwords = stopwords.words('english')

classifier = train_sk_rfc(input_file)
save_sklearn_to_onnx(classifier, "testclf72020.onnx")
# train_models(input_file, test_file)

sess = load_model_onnx("testclf72020.onnx")
# sess = load_model_onnx("model.onnx")

# sess = load_model_onnx('~/mldata/unbalanced4.6.onnx')

# output_file = "unbalanced4.6_result1.csv"
# predict_and_merge(test_file, output_file, sess)

# test_file = "ml data - full - biased - testSet1.csv"

sample = "signed and dated 🔥ends when I call it 💎💎Starts at $5 and $5 usd min increments ( no reserve ) 🔥🔥Free shipping in the US ($10 usd to Mexico/ Canada ) 🔥💎please tag the person you outbid 💎failure to pay within 24 hours of winning auction or erasing bids = block 💎 thanks for all the support. Good luck 🍀👍🏻 #rasetglass #glassart #glassauction  #glassofig #glass #glassofig #glassforsale #glassart #glassblowing #glass_of_ig #pendysofig #pendys"
# sample = 'Custom hand burned Shogun display box!!!💮🏯🎏🎋⛩ NFS #woodburning #colorado #boulder #woodencass #pine #woodwork #wood #japeneses #scarab #shogun #satisfying #woodart #handcarved #japeneseglass #woodworking #woodcarving #workshop #bestofglass #love #pin #katakana #woodcase #case #displaycase #engraving #dremel #woodartist #headyart #japanesestyle #srg2019'
# sample = pre_proc_text(sample)
predict([sample], sess)

# test_file = "testSet.csv"
    # with open(test_file, 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         predict_tf_keras(row[2], sess)
