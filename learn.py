import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from time import time
import pickle

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer


def train_model():
    input_file = "train2.csv"
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
        max_features=10, min_df=5, max_df=0.7,
        stop_words=stopwords.words('english'))
    x = vectorizer.fit_transform(documents).toarray()
    print(str(time() - start))

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
    test_size=0.2, random_state=0)

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(
        n_estimators=1000, random_state=0)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix, accuracy_score
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

    # Convert to ONNX format 
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(classifier, initial_types=initial_type)
    with open("rf_iris.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    
    # import onnxruntime as rt
    # sess = rt.InferenceSession("rf_iris.onnx")
    # input_name = sess.get_inputs()[0].name
    # label_name = sess.get_outputs()[0].name
    # pred_onx = sess.run([label_name], {input_name: x_test.astype(np.float32)})[0]


# vectorizer = TfidfVectorizer(
#         max_features=10, min_df=5, max_df=0.7,
#         stop_words=stopwords.words('english'))

# SAVE MODEL
# with open("text_classifier", "wb") as picklefile:
#     pickle.dump(classifier, picklefile)
# Load Model

# with open("text_classifier", "rb") as training_model:
#     model = pickle.load(training_model)

def merge_files():
    input_file = "tests2.csv"
    df = pd.read_csv(input_file, header=None)
    df.columns = ["data", "target"]

    df, y = df.data, df.target

    test_documents = []
    for caption in range(0, len(df)):
        document = str(df[caption])
        test_documents.append(document)

    x = vectorizer.fit_transform(test_documents).toarray()

    prediction = pd.DataFrame(model.predict(x))
    print(prediction)

    np.savetxt('result.csv', prediction, delimiter=',')
    data = pd.read_csv("tests2.csv")
    data[3] = prediction
    data.to_csv("output.csv", header=False)

train_model()

output_file = "output.csv"
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