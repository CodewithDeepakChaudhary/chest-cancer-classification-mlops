
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:

    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        model = load_model(os.path.join("model", "model.h5"))

        test_image = image.load_img(
            self.filename,
            target_size=(224, 224)
        )

        test_image = image.img_to_array(test_image)

        # IMPORTANT: Normalize image just like training
        test_image = test_image / 255.0

        test_image = np.expand_dims(test_image, axis=0)

        # Get prediction probabilities
        prediction = model.predict(test_image, verbose=0)

        print("Model output:", prediction)

        result = np.argmax(prediction, axis=1)

        print("Predicted class:", result)

        if result[0] == 1:
            prediction_label = "Normal"
        else:
            prediction_label = "Adenocarcinoma Cancer"

        return [{
            "image": prediction_label
        }]
