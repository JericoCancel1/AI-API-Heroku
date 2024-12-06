from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os
import boto3
import tensorflow as tf
import tensorflow_addons as tfa
import zipfile
import logging
from io import BytesIO
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model


app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

# S3 configuration
AWS_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

PICKLE_FILE_KEY = "dataset_embeddingss.pkl"
MODEL_ZIP_FILE_KEY = "siamese_model.h5"
LOCAL_PICKLE_FILE = "/tmp/dataset_embeddingss.pkl"
LOCAL_MODEL_ZIP_FILE = "/tmp/siamese_model.h5"
LOCAL_MODEL_DIR = "/tmp/siamese_model"

# Custom objects for loading the model
custom_objects = {"Addons>TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss}
target_size = (28, 28)


def download_from_s3(s3_key, local_path):
    """Download a file from S3."""
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        s3_client.download_file(AWS_BUCKET_NAME, s3_key, local_path)
        logger.info(f"Downloaded {s3_key} from S3 to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {s3_key}: {e}")
        raise

def extract_zip_file(zip_path, extract_to):
    """Extract a zip file to a specific directory."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted {zip_path} to {extract_to}")
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        raise

def load_model_and_embeddings():
    """Download and load the model and embeddings from S3."""
    # Download and extract the model
    download_from_s3(MODEL_ZIP_FILE_KEY, LOCAL_MODEL_ZIP_FILE)
    siamese_model = load_model(LOCAL_MODEL_ZIP_FILE, custom_objects=custom_objects)

    # extract_zip_file(LOCAL_MODEL_ZIP_FILE, LOCAL_MODEL_DIR)
    # embedding_model = tf.keras.models.load_model(LOCAL_MODEL_DIR, custom_objects=custom_objects)
    embedding_model = siamese_model.layers[3]
    logger.info("Embedding model loaded successfully.")

    # Download and load the pickle file
    download_from_s3(PICKLE_FILE_KEY, LOCAL_PICKLE_FILE)
    with open(LOCAL_PICKLE_FILE, "rb") as f:
        data = np.load(f, allow_pickle=True)
    dataset_embeddings, filenames = data["embeddings"], data["filenames"]
    logger.info(f"Loaded {len(filenames)} embeddings from pickle file.")
    return embedding_model, dataset_embeddings, filenames

def preprocess_image(file_or_path, target_size):
    """
    Preprocess the image for the model.
    
    :param file_or_path: Path to the image file or an io.BytesIO object.
    :param target_size: Tuple specifying the target size (height, width).
    :return: Preprocessed image as a NumPy array.
    """
    try:
        if isinstance(file_or_path, BytesIO):
            # Load the image directly from BytesIO
            image = load_img(file_or_path, target_size=target_size, color_mode="rgb")
        else:
            # Load the image from a file path
            image = load_img(file_or_path, target_size=target_size, color_mode="rgb")

        # Convert the image to a NumPy array and normalize
        image_array = img_to_array(image) / 255.0
        # Add a batch dimension
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise

def find_closest_embedding(target_embedding, dataset_embeddings, filenames, k=5):
    """Find the closest embeddings to the target embedding."""
    distances = np.linalg.norm(dataset_embeddings - target_embedding, axis=1)
    closest_indices = np.argsort(distances)[:k]
    return [(filenames[idx], distances[idx]) for idx in closest_indices]

@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Handle image uploads and find the closest embedding."""
    try:
        # Validate uploaded image
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        image_file = request.files["image"]

        # Load the model and embeddings
        logger.info("Loading model and embeddings...")
        embedding_model, dataset_embeddings, filenames = load_model_and_embeddings()

        # Convert FileStorage to BytesIO for preprocessing
        logger.info("Preprocessing uploaded image...")
        image_stream = BytesIO(image_file.read())  # Read the file into a BytesIO stream
        preprocessed_image = preprocess_image(image_stream, target_size)

        # Compute the embedding for the uploaded image
        logger.info("Computing embedding for uploaded image...")
        target_embedding = embedding_model.predict(preprocessed_image)

        # Find closest embeddings
        logger.info("Finding closest embeddings...")
        closest_matches = find_closest_embedding(target_embedding, dataset_embeddings, filenames)
        results = [{"filename": fname, "distance": float(distance)} for fname, distance in closest_matches]

        logger.info(f"Results JSON: {results}")


        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error in upload_image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)