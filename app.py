import numpy as np
import tensorflow as tf
import json
import os
import tempfile
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import boto3
import logging
import requests
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
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

# File paths for S3 and local storage
EMBEDDINGS_FILE_KEY = "anchor_embeddings.npy"
MODEL_FILE_KEY = "siamese_model.h5"
LOCAL_EMBEDDINGS_FILE = "/tmp/anchor_embeddings.npy"
LOCAL_MODEL_FILE = "/tmp/siamese_model.h5"
target_size = (28, 28)

# Download file from S3
def download_from_s3(s3_key, local_path):
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

# Load model and embeddings
def load_model_and_embeddings():
    # Download model
    download_from_s3(MODEL_FILE_KEY, LOCAL_MODEL_FILE)
    siamese_model = tf.keras.models.load_model(LOCAL_MODEL_FILE, compile=False)
    embedding_model = siamese_model.layers[3]
    logger.info("Embedding model loaded successfully.")

    # Download and load embeddings
    download_from_s3(EMBEDDINGS_FILE_KEY, LOCAL_EMBEDDINGS_FILE)
    try:
        data = np.load(LOCAL_EMBEDDINGS_FILE, allow_pickle=True)
        filenames = data["filename"]
        embeddings = np.array([np.array(e) for e in data["embedding"]])
        logger.info(f"Loaded {len(filenames)} embeddings from .npy file.")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise

    return embedding_model, embeddings, filenames

# Preprocess image
def preprocess_image(file_path, target_size):
    """
    Preprocess the image stored in a file path.
    :param file_path: Path to the image file.
    :param target_size: Tuple specifying the target size (height, width).
    :return: Preprocessed image as a NumPy array.
    """
    try:
        image = tf.keras.utils.load_img(file_path, target_size=target_size)
        image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        raise

# Find closest embeddings
def find_closest_embeddings(target_embedding, dataset_embeddings, filenames, k=5):
    similarities = cosine_similarity(target_embedding, dataset_embeddings)[0]
    closest_indices = np.argsort(similarities)[-k:][::-1]  # Sort by similarity descending
    return [{"filename": filenames[idx], "similarity": float(similarities[idx])} for idx in closest_indices]

def generate_presigned_url(bucket_name, object_key, expiration=3600):
	s3 = boto3.client('s3', region_name='us-east-2') 
	try:
		url = s3.generate_presigned_url( 
			ClientMethod='get_object', 
			Params={'Bucket': bucket_name, 'Key': object_key}, 
			ExpiresIn=expiration 
		) 
		print(f"Generated URL: {url}") 
		return url 
	except Exception as e: 
		logging.error(f"Error generating pre-signed URL: {str(e)}") 
		return None


def download_image_from_url(url, local_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        # Send an HTTP GET request to fetch the image
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Write the content to a local file
        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Image downloaded successfully to {local_path}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download image: {e}")

def process_image(filename):
    try:
        bucket_name = "capstone-bucket-heroku"
        object_key = f"anchor/{filename}"

        # Generate the pre-signed URL
        logger.info(f"Generating pre-signed URL for: {object_key}")
        url = generate_presigned_url(bucket_name, object_key)

        if url:
            download_image_from_url(url, "tmp/" + filename)
            processed_image = Image.open("tmp/" + filename)
            # processed_image = img.convert("RGB")
            logger.info(f"Processed image loaded successfully: {object_key}")
            processed_image = processed_image.convert("RGB")
            return processed_image
        else:
            raise Exception("Failed to generate pre-signed URL")
    except Exception as e:
        logger.error(f"Error in process_image: {e}")
        raise

@app.route("/api/upload", methods=["POST"])
def upload_image():
    try:
        # Validate uploaded image
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        image_file = request.files["image"]

        # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp_file:
            tmp_file_path = tmp_file.name
            image_file.save(tmp_file_path)
        logger.info(f"Saved uploaded image to temporary file: {tmp_file_path}")

        # Load model and embeddings
        logger.info("Loading model and embeddings...")
        embedding_model, dataset_embeddings, filenames = load_model_and_embeddings()

        # Preprocess uploaded image
        logger.info("Preprocessing uploaded image...")
        preprocessed_image = preprocess_image(tmp_file_path, target_size)
        logger.info(f"Preprocessed image shape: {preprocessed_image.shape}")

        # Compute embedding for uploaded image
        logger.info("Computing embedding for uploaded image...")
        target_embedding = embedding_model.predict(preprocessed_image)
        logger.info(f"Computed embedding: {target_embedding}")

        # Find closest embeddings
        logger.info("Finding closest embeddings...")
        closest_matches = find_closest_embeddings(target_embedding, dataset_embeddings, filenames)

        # Process the closest match image
        logger.info("Processing the closest match image...")
        closest_match_filename = closest_matches[1]['filename']  # Use the closest match
        processed_image = process_image(closest_match_filename)

        img_io = BytesIO()
        processed_image.save(img_io, format="PNG")
        img_io.seek(0)


        # Remove the temporary file
        os.remove(tmp_file_path)
        logger.info(f"Removed temporary file: {tmp_file_path}")

        # Return the processed image
        return send_file(img_io, mimetype='image/png')

    except Exception as e:
        logger.error(f"Error in upload_image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)