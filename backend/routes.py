from flask import Blueprint, request, jsonify, current_app
import numpy as np
import cv2
import os

main = Blueprint('main', __name__)

# predict route for image classification
@main.route('/predict', methods=['POST'])
def predict():
    # ensure image is part of the request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    # validate if the file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # validate allowed extensions
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    # process the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    # predict using the model
    model = current_app.config['MODEL']
    prediction = model.predict(np.expand_dims(img, axis=0))

    # get predicted class
    predicted_class = np.argmax(prediction, axis=1)

    classes = {0: 'Tumor free', 1: 'Meningioma tumor', 2: 'Glioma tumor', 3: 'Pituitary tumor'}

    return jsonify({'prediction': str(classes[predicted_class[0]])})

# helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']
