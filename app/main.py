from flask import Flask, jsonify, request
import requests

from flair.data import Sentence
from flair.models import SequenceTagger

app = Flask(__name__)

tagger = SequenceTagger.load("flair/pos-english")
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-MiniLM-L6-v2"
headers = {"Authorization": "Bearer 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"}


@app.before_request
def before():
  print('|\n|\n|')


@app.route('/hf/', methods=['POST'])
def sentiment():
  customerRequest = Sentence(request.json['request'])
  categories = request.json['categories']
  tagger.predict(customerRequest)
  text = 'none'
  for entity in customerRequest.get_spans('pos'):
    if (entity.tag == 'NN') | (entity.tag == 'NNS'):
      text = entity.text
  print(text)
  def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
  data = query(
    {
      "inputs": {
        "source_sentence": text,
        "sentences": categories
      }
    }
  )
  resultsWithLabels = set(zip(categories,data))

  bestMatch = sorted(resultsWithLabels, key=lambda x: x[1], reverse=True)[0]
  print(bestMatch)
  return jsonify({"category": bestMatch[0], "item": text})

# sentence = urllib.parse.unquote(sentence)
