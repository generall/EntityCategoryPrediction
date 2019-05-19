"""
REST API Resource Routing
http://flask-restplus.readthedocs.io
"""
import os
from datetime import datetime
from flask import request
from flask_restplus import Resource

from category_prediction.mention_extractor import MentionExtractor
from .security import require_auth
from . import api_rest

model_path = os.environ['MODEL']
mention_extractor = MentionExtractor(model_path)


class SecureResource(Resource):
    """ Calls require_auth decorator on all requests """
    method_decorators = [require_auth]


@api_rest.route('/extract_and_predict')
class ResourceOne(Resource):
    """ Unsecure Resource Class: Inherit from Resource """

    def post(self):

        content = request.json

        if not isinstance(content, dict):
            return {'error': "No content"}, 400

        if 'text' not in content:
            return {'error': f"No text in {content}"}, 400

        if not isinstance(content['text'], str):
            return {'error': "text is not string"}, 400

        results = mention_extractor.extract_and_predict(content['text'])

        response = results

        return response


@api_rest.route('/secure-resource/<string:resource_id>')
class SecureResourceOne(SecureResource):
    """ Unsecure Resource Class: Inherit from Resource """

    def get(self, resource_id):
        timestamp = datetime.utcnow().isoformat()
        return {'timestamp': timestamp}
