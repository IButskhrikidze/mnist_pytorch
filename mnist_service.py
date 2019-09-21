from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, render_template, jsonify
from torchvision import transforms
import predict

app = Flask(__name__)


def init_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def _process_array(img_array):
    img_binary = Image.fromarray(img_array, mode='L')

    return img_binary


def _resize_algorithm(img, dim):
    if img is None:
        rimg = Image.BILINEAR
    elif img.size[0] > dim[0] or img.size[1] > dim[1]:
        rimg = Image.ANTIALIAS
    else:
        rimg = Image.BILINEAR

    return rimg


@app.route("/", methods=['POST', 'GET'])
def mnist_request():
    if request.method == 'GET':
        resp = render_template('index.html')
    elif request.method == 'POST':
        with Image.open(BytesIO(request.data)) as img:
            img = img.convert('L')  # convert into grey-scale
            img = img.point(lambda i: i < 150 and 255)  # better black and white
            img = ImageOps.expand(img, border=1, fill='black')  # add padding
            rimg = _resize_algorithm(img, (28, 28))
            img.thumbnail((28, 28), rimg)  # resize back to the same size
            img = np.array(img)
            img = _process_array(img)
            img.save('./trained_model/input_image.jpg')

            pre = predict.Predict()

            resp = jsonify(pre)

    return resp


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='2223', threaded=True)
