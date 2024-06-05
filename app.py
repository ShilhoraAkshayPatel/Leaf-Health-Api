from flask_cors import CORS
import os
import base64
import numpy as np
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
import tensorflow as tf
import keras
import PIL
from keras import layers
from keras import backend
from keras.applications import MobileNetV2
os.environ["KERAS_BACKEND"] = "tensorflow"

app = Flask(__name__)
cors = CORS(app)

inputs = layers.Input(shape=(224, 224, 3))
model = MobileNetV2(
    include_top=False,
    input_tensor=inputs,
    # weights="imagenet",
    # classifier_activation="softmax",
)

# Rebuild top
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization()(x)

top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(38, activation="softmax", name="pred")(x)

label_smoothing_factor = 0.1
loss_fn = keras.losses.CategoricalCrossentropy(
    label_smoothing=label_smoothing_factor
)

# Compile
model = keras.Model(inputs, outputs, name="MobileNetV2")
optimizer = keras.optimizers.Adam(learning_rate=1e-4)

model.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=["accuracy"],
    jit_compile=False,
)

print(f"==>> model main: {model}")
MODEL = model.load_weights("plant_village_ckpt.weights.h5")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/")
def index():
    return "Api is working go to /api/predict to get predction with img as input"

@app.route("/api/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(224, 224))
    print(f"==>> processed_image: {processed_image.shape}")
    print(f"==>> model: {model}")
    prediction = model.predict(processed_image)
    print(prediction)
    preclass = {
        0: "Apple_scab",
        1: "Apple_black_rot",
        2: "Apple_cedar_apple_rust",
        3: "Apple_healthy",
        4: "Background_without_leaves",
        5: "Blueberry_healthy",
        6: "Cherry_powdery_mildew",
        7: "Cherry_healthy",
        8: "Corn_gray_leaf_spot",
        9: "Corn_common_rust",
        10: "Corn_northern_leaf_blight",
        11: "Corn_healthy",
        12: "Grape_black_rot",
        13: "Grape_black_measles",
        14: "Grape_leaf_blight",
        15: "Grape_healthy",
        16: "Orange_haunglongbing",
        17: "Peach_bacterial_spot",
        18: "Peach_healthy",
        19: "Pepper_bacterial_spot",
        20: "Pepper_healthy",
        21: "Potato_early_blight",
        22: "Potato_healthy",
        23: "Potato_late_blight",
        24: "Raspberry_healthy",
        25: "Soybean_healthy",
        26: "Squash_powdery_mildew",
        27: "Strawberry_healthy",
        28: "Strawberry_leaf_scorch",
        29: "Tomato_bacterial_spot",
        30: "Tomato_early_blight",
        31: "Tomato_healthy",
        32: "Tomato_late_blight",
        33: "Tomato_leaf_mold",
        34: "Tomato_septoria_leaf_spot",
        35: "Tomato_spider_mites_two-spotted_spider_mite",
        36: "Tomato_target_spot",
        37: "Tomato_mosaic_virus",
        38: "Tomato_yellow_leaf_curl_virus"
    }
    response = preclass[np.argmax(prediction)]
    print("response", response)
    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(
        (
            "* Loading Keras model and Flask starting server..."
            "please wait until server has fully started"
        )
    )

    app.run(host="0.0.0.0", port=port)
