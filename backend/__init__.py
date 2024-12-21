from flask import Flask
import tensorflow as tf

def create_app():
    app = Flask(__name__)
    
    # configuration settings
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['RESULTS_FOLDER'] = 'static/results'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
    
    # load the model
    model_path = '/model/model.h5'
    app.config['MODEL'] = tf.keras.models.load_model(model_path)

    # register blueprints
    from routes import main
    app.register_blueprint(main)

    return app