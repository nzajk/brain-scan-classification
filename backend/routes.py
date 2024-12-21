from flask import Blueprint, render_template, request, jsonify
import numpy as np
import cv2
from flask import current_app

# define a blueprint, a way to organize routes and views
main = Blueprint('main', __name__)

@main.route('/')
def home():
    # this route handles the home page request (GET)
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    # this route handles prediction requests
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # process the image and make a prediction
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    
    model = current_app.config['MODEL']
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class = np.argmax(prediction, axis=1) # get the highest probability class

    # return result
    result = {'prediction': str(predicted_class[0])}
    return jsonify(result)
