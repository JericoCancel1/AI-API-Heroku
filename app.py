from flask import Flask, jsonify, request, send_file
from flask_cors import CORS, cross_origin
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa

app = Flask(__name__)
CORS(app)
@app.route('/')
def index():
    return "Flask Heroku App"
# Custom objects required for the model
custom_objects = {"Addons>TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss}
siamese_model = load_model("siamese_model.h5", custom_objects=custom_objects)
embedding_model = siamese_model.layers[3]  # Update this index as per your model

# Load and preprocess the dataset
dataset_path = "D:/dataset"  # Update with your dataset path
target_size = (28, 28)  # Ensure consistency with your model's input size

def preprocess_dataset(dataset_path, target_size):
    dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels=None,  # No labels needed
        color_mode="rgb",
        batch_size=32,
        image_size=target_size
    )
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    dataset = dataset.map(lambda x: normalization_layer(x))
    return dataset

preprocessed_dataset = preprocess_dataset(dataset_path, target_size)
dataset_images = np.concatenate([batch for batch in preprocessed_dataset])
dataset_embeddings = embedding_model.predict(dataset_images)

# Helper function to preprocess a single image
def preprocess_image(image, target_size=(28, 28)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Helper function to find the closest embeddings
def find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images, k=5):
    distances = np.linalg.norm(dataset_embeddings - target_embedding, axis=1)
    closest_indices = np.argsort(distances)[:k]
    return [(dataset_images[idx], distances[idx]) for idx in closest_indices]

# API route to process uploaded image
@app.route('/api/upload', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def upload_image():
    # Check if an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Get the image file
    image_file = request.files['image']
    
    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(image_file, target_size)

    # Generate embedding for the uploaded image
    target_embedding = embedding_model.predict(preprocessed_image)

    # Find the closest embeddings
    closest_images = find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images)

    # Create a response with closest image data
    results = []
    for i, (image_array, distance) in enumerate(closest_images):
        # Convert the image array to a PIL Image for response
        image = Image.fromarray((image_array * 255).astype('uint8'))
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        results.append({"index": i, "distance": float(distance)})

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)