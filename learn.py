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
    caption, tags = extract_hashtags(caption)
    caption, mentions = extract_mentions(caption)
    caption = remove_punct(caption)
    caption = tokenize(caption)
    caption = remove_stopwords(caption, stop_words)

    caption = ' '.join(caption)
    caption = caption.rstrip()
    
    # return [post[0], post[1], caption]
    return caption


# def train_tf_keras(input_file):
    #     from multiprocessing import Pool
    #     start = time()

    #     df = pd.read_csv(input_file, header=0, usecols=["og_caption", "target"])
    #     og_captions = df['og_caption'].values.tolist()

    #     captions = []
    #     pool = Pool(3)
    #     captions = [x for x in pool.map(pre_proc_text, og_captions) if x is not None]
    #     df['clean_caption'] = pd.Series(captions)
    #     df, y = df.clean_caption, df.target

    #     documents = []
    #     for caption in range(0, len(df)):
    #         document = str(df[caption])
    #         documents.append(document)

    #     # df = np.array(df)
    #     # y = np.array(y)

    #     from sklearn.model_selection import train_test_split
    #     sentences_train, sentences_test, y_train, y_test = train_test_split(
    #         df, y, test_size=0.15, random_state=1000)
    #     from keras.preprocessing.text import Tokenizer
    #     tokenizer = Tokenizer(num_words=5000)
    #     tokenizer.fit_on_texts(sentences_train)
    #     pickle.dump(tokenizer, open("tokenizer.pickle", "wb"))
    #     print("saved tokenizer")
    #     X_train = tokenizer.texts_to_sequences(sentences_train)
    #     X_test = tokenizer.texts_to_sequences(sentences_test)
        
    #     vocab_size = len(tokenizer.word_index) + 1
    #     print(sentences_train[2])
    #     print(X_train[2])
        
    #     from keras.preprocessing.sequence import pad_sequences
    #     X_train = pad_sequences(
    #         X_train, padding="post", maxlen=500)
    #     X_test = pad_sequences(X_test, padding="post", maxlen=500)
    #     print(X_train[0, :])

    #     from keras.models import Sequential
    #     from keras import layers
    #     model = Sequential()
    #     print("model = Sequential")
    #     model.add(layers.Embedding(
    #         input_dim=vocab_size,
    #         output_dim=50,
    #         input_length=500))
    #     print("model layers embedding")
    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(10, activation="relu"))
    #     model.add(layers.Dense(1, activation="sigmoid"))
    #     model.compile(
    #         optimizer="adam",
    #         loss="binary_crossentropy",
    #         metrics=["accuracy"])
    #     print("model compiled")
    #     model.summary()
    #     print("summary")
    #     history = model.fit(
    #         X_train, y_train,
    #         epochs=20,
    #         verbose=False,
    #         validation_data=(X_test, y_test),
    #         batch_size=10)
    #     print("fitted model")
    #     loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    #     print("Training accuracy: {:.4f}".format(accuracy))
    #     loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    #     print("Testing Accuracy : {:.4f}".format(accuracy))
    #     pickle.dump(model, open("tf_kerasmodel.pickle", "wb"))

    #     import keras2onnx
    #     import onnx
    #     onnx_model = keras2onnx.convert_keras(model, model.name)

    #     temp_model_file = "model.onnx"
    #     onnx.save_model(onnx_model, temp_model_file)
    #     # sess = onnxruntime.InferenceSession(temp_model_file)


    #     return model


def train_TF_RFC(input_file):
        from multiprocessing import Pool
        start = time()

        df = pd.read_csv(input_file, header=0, usecols=["og_caption", "target"])
        # df.columns = ["data", "target"] id	og_caption	mod_caption	target
        og_captions = df['og_caption'].values.tolist()

        captions = []
        pool = Pool(3)
        captions = [x for x in pool.map(pre_proc_text, og_captions) if x is not None]
        print("num words per sample: ")
        print(str(get_num_words_per_sample(captions)))
        df['clean_caption'] = pd.Series(captions)
        df, y = df.clean_caption, df.target
        # df, y = df.og_caption, df.target

        documents = []
        for caption in range(0, len(df)):
            document = str(df[caption])
            documents.append(document)

        vectorizer = TfidfVectorizer(
            max_features=80, stop_words=stopwords)
        print(str(time() - start))

        X = vectorizer.fit_transform(documents).toarray()
        print(str(time() - start))

        pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
        print(str(time() - start))

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

        from keras.models import Sequential
        from keras import layers
        input_dim = X_train.shape[1]
        model = Sequential()
        model.add(
            layers.Dense(10, input_dim=input_dim, activation="relu"))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"])
        model.summary()

        history = model.fit(
            X_train, y_train,
            epochs=20,
            verbose=False,
            validation_data=(X_test, y_test),
            batch_size=5)

        loss, accuracy = model.evaluate(
            X_train, y_train, verbose=False)
        print("Training accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing accuracy: {:.4f}".format(accuracy))
        # from sklearn.ensemble import RandomForestClassifier as rfc
            # import tflearn
            # from tflearn.estimators import RandomForestClassifier as rfc
            
            # train_results = []
            # test_results = []
            
            # m = rfc(
            #     n_estimators=32, 
            #     max_nodes=45,
            #     n_classes=None,
            #     n_features=None,
            #     metric=None,
            #     graph=None,
            #     global_step=None)

            # m.fit(X_train, y_train)
            # print(m.evaluate(X_train, X_test, tflearn.accuracy_op))
            # print(m.evaluate(y_train, y_test, tflearn.accuracy_op))
            # print(m.predict(y_train))
        
        # train_pred = m.predict(X_train)
        # y_pred = m.predict(X_test)

        print(str(time() - start))
        
        
        with open("tf_rf_classifier", "wb") as picklefile:
            pickle.dump(classifier, picklefile)
        return classifier


def train_sk_rfc(input_file):
    # from stop_words import get_stop_words
    # stopwords = get_stop_words

    df = pd.read_csv(input_file, header=None)
    df.columns = ["data", "target"]

    df, y = df.data, df.target

    documents = []
    for caption in range(0, len(df)):
        document = str(df[caption])
        document = pre_proc_text(document)
        documents.append(document)
    # print(df)
    # Convert to numbers with bag of words

    start = time()
    stop_words = get_stop_words('english')
    
    vectorizer = TfidfVectorizer(
        max_features=80, stop_words=stop_words)
    print(str(time() - start))

    X = vectorizer.fit_transform(documents).toarray()
    print("SHAPE")
    print(X.shape)

    print(str(time() - start))

    pickle.dump(vectorizer, open("vectorizer2020-learnpy.pickle", "wb"))
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

    # y_pred = classifier.predict(X_test)
        # fpr, tpr, thresholds = roc_curve(y_train, train_pred)
        # roc_auc = auc(fpr, tpr)
        # train_results.append(roc_auc)
        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # roc_auc = auc(fpr, tpr)
        # test_results.append(roc_auc)

    y_pred = classifier.predict(X_test)

    # from matplotlib.legend_handler import HandlerLine2D
        # import matplotlib as mpl
        # import matplotlib.pyplot as plt
        # line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
        # line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
        # plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        # plt.ylabel("AUC score")
        # plt.xlabel("min samples leaf")
        # plt.show()

    print(str(time() - start))
    
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print('Mean Absolute Error:', mean_squared_error(y_test, y_pred))
    print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)))
    cross_validate(X_train, y_train, classifier)
    with open("testclf72020-learnpy", "wb") as picklefile:
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

    vectorizer = pickle.load(open("vectorizer2020-learnpy.pickle", "rb"))

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
    vectorizer = pickle.load(open("vectorizer2020-learnpy.pickle", "rb"))
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
   


def cross_validate(X_train, y_train, clf):
    from sklearn.model_selection import cross_val_score
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


input_file = "~/mldata/ml data - full - biased - testSet.csv"

classifier = train_sk_rfc(input_file)
save_sklearn_to_onnx(classifier, "testclf72020-learnpy.onnx")

sess = load_model_onnx("testclf72020-learnpy.onnx")


sample = "signed and dated 🔥ends when I call it 💎💎Starts at $5 and $5 usd min increments ( no reserve ) 🔥🔥Free shipping in the US ($10 usd to Mexico/ Canada ) 🔥💎please tag the person you outbid 💎failure to pay within 24 hours of winning auction or erasing bids = block 💎 thanks for all the support. Good luck 🍀👍🏻 #rasetglass #glassart #glassauction  #glassofig #glass #glassofig #glassforsale #glassart #glassblowing #glass_of_ig #pendysofig #pendys"
# sample = 'Custom hand burned Shogun display box!!!💮🏯🎏🎋⛩ NFS #woodburning #colorado #boulder #woodencass #pine #woodwork #wood #japeneses #scarab #shogun #satisfying #woodart #handcarved #japeneseglass #woodworking #woodcarving #workshop #bestofglass #love #pin #katakana #woodcase #case #displaycase #engraving #dremel #woodartist #headyart #japanesestyle #srg2019'
sample = pre_proc_text(sample)
print(f"sample: {type(sample)}")
sample = [sample]
print(f"sample: {type(sample)}")
print(sample)
predict(sample, sess)

# test_file = "testSet.csv"
    # with open(test_file, 'r') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:
    #         predict_tf_keras(row[2], sess)
