import os
import spacy

from allennlp.models import load_archive
from allennlp.predictors import Predictor
from flask import Flask, jsonify, request


from category_prediction.predictor import PersonPredictor

app = Flask(__name__)

model_path = os.environ['MODEL']

archive = load_archive(model_path, overrides=PersonPredictor.overrides)
predictor = Predictor.from_archive(archive, 'person-predictor')


def check_categories_content(content):
    if 'mentions' not in content:
        return "No 'mentions' found", 400

    mentions = content['mentions']

    if not isinstance(mentions, list):
        return "'mentions' expected to be a list of strings", 400

    if len(mentions) == 0:
        return "Mentions list is empty", 400

    for idx, mention in enumerate(mentions):
        if not isinstance(mention, str):
            return f"mention {idx} is not a {type(mention)}. String expected", 400

        if len(mention) < 10:
            return f"mention {idx} is too small. It has only {len(mention)} chars, min size: {10}", 400

        if len(mention) > 1000:
            return f"mention {idx} is too large. Its size is {len(mention)}, max size: {1000}", 400

    return None


@app.route("/predict_category", methods=['POST'])
def predict_category():
    content = request.get_json()

    if not isinstance(content, dict):
        return jsonify({'error': "No content"}), 400

    check = check_categories_content(content)
    if check is not None:
        error, code = check
        return jsonify({'error': error}), code

    result = predictor.predict_json(content)

    return jsonify(result)




@app.route("/extract_people", methods=['POST'])
def extract_people():
    content = request.get_json()


    return jsonify("Hello World!")
