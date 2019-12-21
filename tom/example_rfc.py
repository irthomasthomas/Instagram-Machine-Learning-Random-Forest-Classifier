import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

corpus = [
    "The apple is on sale apple", "The oranges are on orange sale",
    "The apple and is present", "The orange and is present"]

Y = np.array([0,1,0,1])
vectorizer = CountVectorizer(min_df=1)

X = vectorizer.fit_transform(corpus).toarray()
print(X.shape)
clf = RandomForestClassifier()
clf.fit(X, Y)

sample = vectorizer.transform(
        ["orange is present tomorrow this friday"]).toarray()
print(sample.shape)
pred = clf.predict(sample)
print(pred)

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
initial_type = [("float_input", FloatTensorType([1, 10]))]
onx = convert_sklearn(clf, initial_types=initial_type)
with open("sample_rfc_onx.onnx", "wb") as f:
        f.write(onx.SerializeToString())

import onnxruntime as rt
sess = rt.InferenceSession("sample_rfc_onx.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
label_name2 = sess.get_outputs()[1].name

print(input_name)
print(label_name)
print(label_name2)

pred_onx = sess.run(
    [label_name], {input_name: sample.astype(np.float32)})[0]
print(pred_onx)


import redis
conn = redis.Redis(host="localhost", port=6379, db=0)
with open("sample_rfc_onx.onnx", "rb") as f:
    model = f.read()
    res = conn.execute_command('AI.MODELSET', 'sklmodel', 'ONNX', 'CPU', model)
    print(res)

import redisai

rai = redisai.Client()
tensor = redisai.BlobTensor.from_numpy(
    np.ones((1, 10), dtype=np.float32))
tensor = redisai.BlobTensor.from_numpy(
    sample.astype(np.float32))
rai.tensorset('tensor', tensor)
# conn.modelset('model', Backend.onnx, device, model)
rai.modelrun('sklmodel', inputs=['tensor'], outputs=['out_label', 'out_probs'])
out = rai.tensorget('out_label')
# out_probs = rai.tensorget('out_probs')

# out = con.tensorget('out', as_type=BlobTensor)
print(out)
# print(out_probs.to_numpy())