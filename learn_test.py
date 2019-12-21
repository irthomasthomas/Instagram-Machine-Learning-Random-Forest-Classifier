import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

with open("text_classifier", "rb") as training_model:
    model = pickle.load(training_model)

x = " sold dibs alert first dibs snag beauty 25 free shipping us 12 international must provide id artist "
x = x.astype(np.float64)
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
x = tfidfconverter.fit_transform(x).toarray()
# vec = CountVectorizer()
# X = vec.fit_transform(sample)
# clf = RandomForestClassifier()  
# clf.fit(X, y) 

# vectorizer = TfidfVectorizer(
#         max_features=10, min_df=5, max_df=0.7,
#         stop_words=stopwords.words('english'))
# x = vectorizer.fit_transform(sample).toarray()

y_pred2 = model.predict([x])
print(y_pred2)
