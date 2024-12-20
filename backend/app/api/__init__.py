from flask import Flask
from tensorflow.keras.models import load_model
import os

def create_app():
    app = Flask(__name__)
    
    # configuration settings
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['RESULTS_FOLDER'] = 'static/results'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    
    # load the model
    model_path = 'backend/app/model/model.h5'
    app.config['MODEL'] = load_model(model_path)

    # register blueprints
    from backend.app.api.routes import main
    app.register_blueprint(main)

    return app
