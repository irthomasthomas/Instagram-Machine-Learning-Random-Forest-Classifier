import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from time import time
import pickle
import csv

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

def train_rfc(input_file):
    df = pd.read_csv(input_file, header=None)
    df.columns = ["data", "target"]

    df, y = df.data, df.target

    documents = []
    for caption in range(0, len(df)):
        document = str(df[caption])
        documents.append(document)
    print(df)
    # Convert to numbers with bag of words
    """ start = time()
    from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(
            max_features=10, min_df=5, max_df=0.7,
            stop_words=stopwords.words('english'))
        x = vectorizer.fit_transform(documents).toarray()
        # BOW assigns score to words based on document context
        # To resolve this, multiply the term freqeuncy by the inverse
        # ...document frequency
        # TF = (Number of Occurrences of a word)/(Total words in the document)
        # IDF(word) = Log((Total number of documents)/(Number of documents
        # containing the word))
        from sklearn.feature_extraction.text import TfidfTransformer
        tfidfconverter = TfidfTransformer()
        x = tfidfconverter.fit_transform(x).toarray()
        print(str(time() - start))
        print(str(x)) """

    start = time()
    vectorizer = TfidfVectorizer(
        max_features=80, stop_words=stopwords.words('english'))
    print(str(time() - start))

    X = vectorizer.fit_transform(documents).toarray()
    pickle.dump(vectorizer, open("vectorizer.pickle", "wb"))
    print(str(time() - start))

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

    from sklearn.ensemble import RandomForestClassifier as rfc
    
    train_results = []
    test_results = []

    # for max in max_features:
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

    y_pred = classifier.predict(X_test)

        # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        # roc_auc = auc(fpr, tpr)
        # test_results.append(roc_auc)

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
    # print('Mean Absolute Error:', mean_squared_error(y_test, y_pred))
    # print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)))
    # cross_validate(X_train, y_train, classifier)
    with open("text_classifier", "wb") as picklefile:
        pickle.dump(classifier, picklefile)
    return classifier


def save_to_onnx(classifier, output_filename):
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
    vectorizer = pickle.load(open("vectorizer.pickle", "rb"))
    input_name = onx.get_inputs()[0].name
    label_name = onx.get_outputs()[0].name
    print(input_name)
    print(label_name)
    # input = [text,]
    X = vectorizer.transform(text).toarray()
   
    pred_onx = onx.run(
        [label_name], {input_name: X.astype(np.float32)})[0]
    print(pred_onx)
   
    
    with open("text_classifier", "rb") as training_model:
        model = pickle.load(training_model)

    prediction = model.predict(X)
    print(str(prediction))


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


# input_file = "ml data - full - balanced.csv"
input_file = "data/ml data - full - biased.csv"
# input_file = "mlOutput - testSet.csv"
# input_file = "ml data - full - biased - testSet.csv"
test_file = "ml data - full - biased - testSet1.csv"
# classifier = train_rfc(input_file)
# save_to_onnx(classifier, "unbalanced4.5.onnx")
    # train_models(input_file, test_file)
sess = load_model_onnx("unbalanced4.5.onnx")
output_file = "unbalanced4.5_result2.csv"
# predict_and_merge(test_file, output_file, sess)

with open(test_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        predict(row, sess)


# SAVE MODEL
# with open("text_classifier", "wb") as picklefile:
#     pickle.dump(classifier, picklefile)

# Load Model
# with open("text_classifier", "rb") as training_model:
#     model = pickle.load(training_model)

""" import csv
    import itertools as IT
    filenames = ["testest.csv", "result.csv"]
    handles = [open(filename, 'r') for filename in filenames]
    readers = [csv.reader(f, delimiter=",") for f in handles]

    with open(output_file, "w", newline="") as outputfile:
        writer = csv.writer(outputfile, delimiter=",", lineterminator="\n")
        for rows in IT.zip_longest(*readers, fillvalue=[""]*2):
            combined_row = []
            for row in rows:
                row = row[:2]
                if len(row) == 2:
                    combined_row.extend(row)
                else:
                    combined_row.extend([""]*2)
            writer.writerow(combined_row)
    for f in handles:
        f.close() """