import numpy as np
import io
import tensorflow as tf
from PIL import Image
from services.configuration import get_settings


def read_file_as_image(file) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file))
        return np.array(image)
    except Exception as e:
        raise e.add_note(
            "Error reading file as image. Please ensure it is a valid image file."
        )


class Model:
    _instance = None

    @staticmethod
    def get_model():
        try:
            settings = get_settings()
            if Model._instance is None:
                Model._instance = tf.keras.models.load_model(
                    settings.MODEL_PATH)
            return Model._instance
        except Exception as e:
            raise e.add_note(
                "Error loading model. Please ensure the model exists at the specified path."
            )

    @staticmethod
    def predict(input_data):
        try:
            settings = get_settings()
            model = Model.get_model()
            prediction = model.predict(input_data)
            confidence = round(float(max(prediction[0])), 2)
            label = settings.CLASS_NAMES[np.argmax(prediction[0])]
            return label, confidence
        except Exception as e:
            raise e.add_note(
                "Error while predicting label"
            )
