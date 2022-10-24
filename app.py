import numpy as np
from flask import Flask, jsonify, request, render_template

import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [int(x) for x in request.form.values()]
    final_features = [np.array(features)]
    pred = model.predict(final_features)
    output = pred[0]
    p = output

    if(p == 0):
        text = "Not "
    elif(p == 1):
        text = "Pre "
    else:
        text = "Sim "

    return render_template("index.html", prediction_text="DIAGNOSTICO: " + text + "procure um medico ou uma unidade de saude.")

@app.route("/api", methods=["POST"])
def results():
    data = request.get_json(force=True)
    pred = model.predict([np.array(list(data.values()))])

    output = pred[0]
    return jsonify(output)
