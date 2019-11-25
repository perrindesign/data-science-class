from flask import Flask, escape, request, jsonify
import pickle as pkl
import pandas as pd


# need to train and pickle classifier
with open('./titanic_clf.pkl','rb') as f:
    clf = pkl.load(f)

app = Flask(__name__)

@app.route('/',methods=['POST'])
def predict():

    prediction = None
    print(request)
    req_data = request.get_json()
    print(type(req_data))
    if req_data:
        X = pd.DataFrame(req_data)
        print(X)
        prediction = clf.predict(X)

    return jsonify([str(x) for x in prediction])
