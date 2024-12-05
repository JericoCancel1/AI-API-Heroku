from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from io import BytesIO
from PIL import Image
import numpy as np
import os
import boto3
import tensorflow as tf
import tensorflow_addons as tfa

app = Flask(__name__)
CORS(app)

# S3 configuration
MODEL_FILE_KEY = "siamese_model.h5"  # S3 key for the model file
FOLDER_PREFIX = "tables-20241203T011356Z-001/tables/anchor"  # S3 prefix for the folder
MODEL_LOCAL_PATH = "/tmp/siamese_model.h5"  # Local path to store the model
FOLDER_LOCAL_DIR = "/tmp/tables"  # Local directory to store the folder


def setup():
    """Download necessary files when the app starts."""
    bucket_name = os.getenv("S3_BUCKET_NAME")
    download_file(bucket_name, MODEL_FILE_KEY, MODEL_LOCAL_PATH)
    download_folder(bucket_name, FOLDER_PREFIX, FOLDER_LOCAL_DIR)
    print("Setup complete. Model and tables downloaded.")


def download_file(bucket_name, s3_key, local_path):
    """Download a file from S3."""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        print(f"Downloaded {s3_key} to {local_path}")
    except Exception as e:
        print(f"Failed to download {s3_key} from bucket {bucket_name}: {e}")
        raise


def download_folder(bucket_name, s3_prefix, local_dir):
    """Download all files from an S3 folder."""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    os.makedirs(local_dir, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                file_name = os.path.join(local_dir, obj["Key"].split("/")[-1])
                download_file(bucket_name, obj["Key"], file_name)
        print(f"All files downloaded from {s3_prefix} to {local_dir}")
    except Exception as e:
        print(f"Error downloading folder from S3: {e}")
        raise


def load_models():
    """Load the TensorFlow Siamese model."""
    try:
        custom_objects = {"Addons>TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss}
        print(f"Loading model from {MODEL_LOCAL_PATH}...")
        siamese_model = tf.keras.models.load_model(MODEL_LOCAL_PATH, custom_objects=custom_objects)
        embedding_model = siamese_model.layers[1]  # Adjust based on your architecture
        print("Model loaded successfully.")
        return siamese_model, embedding_model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


def preprocess_image(image, target_size=(28, 28)):
    """Preprocess the uploaded image."""
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images, k=5):
    """Find the closest embeddings to the target embedding."""
    distances = np.linalg.norm(dataset_embeddings - target_embedding, axis=1)
    closest_indices = np.argsort(distances)[:k]
    return [(dataset_images[idx], distances[idx]) for idx in closest_indices]


@app.route("/")
def index():
    return "Flask Heroku App"


@app.route("/api/upload", methods=["POST"])
@cross_origin(origin="*", headers=["Content-Type", "Authorization"])
def upload_image():
    """Handles image uploads, processes them, and returns matching results."""
    _, embedding_model = load_models()

    # Use the global folder path for the dataset
    dataset_path = FOLDER_LOCAL_DIR
    target_size = (28, 28)

    # Load dataset (example placeholder for dataset embedding loading)
    dataset_images = []  # Placeholder
    dataset_embeddings = []  # Placeholder

    # Check if an image is uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Preprocess the uploaded image
    image_file = request.files["image"]
    preprocessed_image = preprocess_image(image_file, target_size)

    # Compute the embedding for the uploaded image
    target_embedding = embedding_model.predict(preprocessed_image)

    # Find the closest matches
    closest_images = find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images)

    # Prepare response with matching results
    results = []
    for i, (image_array, distance) in enumerate(closest_images):
        image = Image.fromarray((image_array * 255).astype("uint8"))
        img_io = BytesIO()
        image.save(img_io, "PNG")
        img_io.seek(0)
        results.append({"index": i, "distance": float(distance)})

    return jsonify({"results": results})


if __name__ == "__main__":
    print("Initializing resources...")
    setup()
    app.run(debug=False)  # Disable debug mode for production
    #djfhaldbflasj