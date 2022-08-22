from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def main():
    return render_template('index.html')


@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    res = preprocessing(file)
    return json.dumps({"image": res.tolist()})


@app.route('/model')
def model():
    json_data = json.load(open("./model_js/model.json"))
    return jsonify(json_data)


@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('model_js', path)


def preprocessing(file):
    img = np.array(Image.open(file), dtype='float64')
    print(type(img))
    img /= 127.5
    img -= 1
    res = cv2.resize(img, (160, 160))
    return res


if __name__ == "__main__":
    app.run(port=5000)
