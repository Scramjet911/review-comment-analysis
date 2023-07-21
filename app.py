from flask import Flask, request, jsonify, Response
from detoxify import Detoxify
from transformers import pipeline

from multiprocessing import Process, Queue

app = Flask(__name__)

sentimentClassifier = pipeline("sentiment-analysis")
abusiveClassifier = Detoxify("original")
sentimentProcess = None
abusiveProcess = None
sentimentQueue = Queue()
abusiveQueue = Queue()


def sentiment_worker(queue):
    while True:
        # Wait for input from the queue
        inputData = queue.get()

        # Perform the ML inference on inputData
        result = sentimentClassifier(input)

        # Put the result in the queue for Flask to retrieve
        queue.put(result)


def abusive_worker(queue):
    while True:
        # Wait for input from the queue
        inputData = queue.get()

        # Perform the ML inference on inputData
        result = sentimentClassifier(input)

        # Put the result in the queue for Flask to retrieve
        queue.put(result)


# If the processes are not running, start them
def process_restart():
    if sentimentProcess is None or not sentimentProcess.is_alive():
        sentimentProcess = Process(
            target=sentiment_worker(queue), args=(sentimentQueue,)
        )
        sentimentProcess.start()

    if abusiveProcess is None or not abusiveProcess.is_alive():
        abusiveProcess = Process(target=sentiment_worker(queue), args=(abusiveQueue,))
        abusiveProcess.start()


@app.route("/", methods=["POST"])
def inference_engine():
    # Get the input data from the POST request
    inputData = request.get_json()["modelInput"]

    process_restart()

    # Pass the input data to the processes through the queue
    sentimentQueue.put(inputData)
    abusiveQueue.put(inputData)

    # Wait for the result from the processes
    abusiveResult = abusiveQueue.get()
    sentimentResult = sentimentQueue.get()

    # Return the result as a response
    print(sentimentResult, flush=True)
    print(abusiveResult, flush=True)

    return jsonify(
        {"sentimentAnalysis": sentimentResult, "abuseAnalysis": abusiveResult}
    )


@app.route("/", methods=["GET"])
def root():
    return """Send a post request with body : 
        {
            modelInput: <String array>
        }

        Response format : 
        {
            abusiveAnalysis:  {
                "identity_attack": [
                    Int
                ],
                "insult": [
                    Int
                ],
                "obscene": [
                    Int
                ],
                "severe_toxicity": [
                    Int
                ],
                "threat": [
                    Int
                ],
                "toxicity": [
                    Int
                ]
            },
            sentimentAnalysis:  {
                "label": "POSITIVE/NEGATIVE",
                "score": Int
            }
        }
    """


if __name__ == "__main__":
    ### If need to load local model
    # model = TFAutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    # classifier([input])

    app.run(debug=True, host="0.0.0.0")
