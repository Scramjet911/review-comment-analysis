from flask import Flask, request, jsonify, Response
from detoxify import Detoxify
import numpy as np
from transformers import pipeline

import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from multiprocessing import Process, Queue
import sys

app = Flask(__name__)

sentimentClassifier = pipeline("sentiment-analysis")
abusiveClassifier = Detoxify("original")


def getSentimentAnalysis(input, q1):
    q1.put(sentimentClassifier(input))
    return 1


def getAbuseAnalysis(input, q2):
    q2.put(abusiveClassifier.predict(input))
    return 1


@app.route("/", methods=["GET", "POST"])
def makeCalc():
    if request.method == "POST":
        data = request.get_json()

        sentimentProcess = Process(
            target=getSentimentAnalysis, args=(data["modelInput"], sentimentQueue)
        )
        abusiveProcess = Process(
            target=getAbuseAnalysis, args=(data["modelInput"], abusiveQueue)
        )

        sentimentProcess.start()
        abusiveProcess.start()

        abusiveResult = abusiveQueue.get()
        sentimentResult = sentimentQueue.get()

        print(sentimentResult, flush=True)
        print(abusiveResult, flush=True)

        return jsonify(
            {"sentimentAnalysis": sentimentResult, "abuseAnalysis": abusiveResult}
        )

    return Response(
        jsonify({error: "Not a proper request method or data"}),
        status=400,
        mimetype="application/json",
    )


if __name__ == "__main__":
    ### If need to load local model
    # model = TFAutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    # classifier([input])

    sentimentQueue = Queue()
    abusiveQueue = Queue()

    app.run(debug=True, host="0.0.0.0")
