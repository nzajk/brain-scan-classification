from flask import Flask
from flask_cors import CORS
from routes import main
import tensorflow as tf

def create_app():
    app = Flask(__name__)
    CORS(app)

    # configuration settings
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['RESULTS_FOLDER'] = 'static/results'
    app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

    # load the model
    model_path = 'backend/model/model.keras'
    app.config['MODEL'] = tf.keras.models.load_model(model_path)

    # register the blueprint for routes
    app.register_blueprint(main)

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
