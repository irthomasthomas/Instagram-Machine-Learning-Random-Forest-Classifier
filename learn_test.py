import pickle

with open("text_classifier", "rb") as training_model:
    model = pickle.load(training_model)

y_pred2 = model.predict([" sold dibs alert first dibs snag beauty 25 free shipping us 12 international must provide id artist "])
print(y_pred2)
