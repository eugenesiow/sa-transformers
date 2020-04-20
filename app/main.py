# -*- coding: utf-8 -*-
""" DSO Sentiment Analysis Transformers

This module contains the application server.
It uses Uvicorn and Starlette for high performance in production systems.

Created by:
    DSO National Laboratories
    Eugene Siow

Routes:
    * `/` Returns the API details.
    * `/sentiment` Returns a json object of the targets, target classes and confidence from POST input text.
"""

import json
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from enum import Enum
from satransformers import SAClassifier


class SARunner:
    def __init__(self, use_cuda=False, cuda_device=0):
        self.model = SAClassifier('roberta', 'model/', use_cuda=use_cuda, cuda_device=cuda_device)
        self.labels = {0: -1, 1: 0, 2: 1}

    def get_sentiment(self, input_list):
        output_list = []
        predictions = self.model.predict(input_list)
        for prediction in predictions:
            output_list.append(self.labels[int(prediction)])
        return output_list

    def shutdown(self):
        del self.model
        print('Shutting down...')


class Messages(Enum):
    INVALID_INPUT = JSONResponse({'ERROR': "Please provide a valid JSON. "}, status_code=400)


DEBUG = False
API_NAME = 'Sentiment Analysis Transformers API'
VERSION = '1.0'

app = Starlette(debug=DEBUG)
sa_runner = SARunner(use_cuda=False, cuda_device=0)


@app.route('/')
async def homepage(_):
    """
    Returns the API details.

    Returns:
        A JSON response with the name and version of the API.
    """
    return JSONResponse({'name': API_NAME, 'version': VERSION})


@app.route('/sentiment', methods=['POST'])
async def sentiment(request):
    """
    Returns a json array list of the sentiment values corresponding to the POST input array list.
    0 is negative, 1 is neutral and 2 is positive.

    Args:
        request: The request object containing a JSON object with the
            parameter 'text' containing the text content to extract targets from.

    Returns:
        Returns a json array list of the sentiment values corresponding to the POST input array list.
        0 is negative, 1 is neutral and 2 is positive.

    Raises:
        JSONDecodeError: The data being deserialized is not valid JSON.
    """
    try:
        req = await request.json()
        if req and isinstance(req, list):
            predictions = sa_runner.get_sentiment(req)
            return JSONResponse(predictions)
        return Messages.INVALID_INPUT.value
    except json.decoder.JSONDecodeError:
        return Messages.INVALID_INPUT.value


@app.on_event('shutdown')
async def shutdown():
    sa_runner.shutdown()

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
