from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from io import BytesIO
from PIL import Image
import numpy as np
import os
import boto3
import tensorflow as tf
import tensorflow_addons as tfa
import zipfile

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
    os.makedirs(FOLDER_LOCAL_DIR, exist_ok=True)  # Ensure the directory exists
    download_file(bucket_name, MODEL_FILE_KEY, MODEL_LOCAL_PATH)
    extract_model(MODEL_LOCAL_PATH, MODEL_EXTRACT_PATH)
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
        files_downloaded = 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                file_name = os.path.join(local_dir, obj["Key"].split("/")[-1])
                download_file(bucket_name, obj["Key"], file_name)
                files_downloaded += 1
        print(f"Downloaded {files_downloaded} files from {s3_prefix} to {local_dir}")
    except Exception as e:
        print(f"Error downloading folder from S3: {e}")
        raise


def load_models():
    """Load the TensorFlow Siamese model saved in SavedModel format."""
    bucket_name = os.getenv("S3_BUCKET_NAME")
    model_key = "siamese_model.zip"  # Update this to your zipped model file in S3

    # Create an S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    try:
        # Download the model .zip file
        print(f"Downloading model {model_key} from bucket {bucket_name}...")
        model_object = s3.get_object(Bucket=bucket_name, Key=model_key)
        model_data = model_object["Body"].read()

        # Save the .zip file locally
        zip_path = "/tmp/siamese_model.zip"
        with open(zip_path, "wb") as zip_file:
            zip_file.write(model_data)

        # Extract the .zip file
        extract_path = "/tmp/siamese_model"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Load the model from the extracted directory
        custom_objects = {"Addons>TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss}
        siamese_model = tf.keras.models.load_model(extract_path, custom_objects=custom_objects)

        # Extract the embedding model (adjust based on your architecture)
        embedding_model = siamese_model.layers[3]  # Adjust this index as needed
        print("Model loaded successfully.")
        return siamese_model, embedding_model

    except Exception as e:
        print(f"Error loading model from S3: {e}")
        raise


def preprocess_image(image_path, target_size=(28, 28)):
    """Preprocess the uploaded image."""
    img = Image.open(image_path).convert("RGB")  # Ensure all images are RGB
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images, k=5):
    """Find the closest embeddings to the target embedding."""
    if len(dataset_embeddings) == 0:
        raise ValueError("Dataset embeddings are empty. Ensure the dataset is processed properly.")
    
    distances = np.linalg.norm(dataset_embeddings - target_embedding, axis=1)
    closest_indices = np.argsort(distances)[:k]
    return [(dataset_images[idx], distances[idx]) for idx in closest_indices]


def compute_dataset_embeddings(dataset_path, embedding_model, target_size):
    """Compute embeddings for the dataset using the embedding model."""
    dataset_images = []
    dataset_embeddings = []
    processed_files = 0

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith((".jpeg", ".jpg", ".png")):
                image_path = os.path.join(root, file)
                try:
                    img = preprocess_image(image_path, target_size)
                    embedding = embedding_model.predict(img)

                    dataset_images.append(np.squeeze(img))
                    dataset_embeddings.append(np.squeeze(embedding))
                    processed_files += 1
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    print(f"Processed {processed_files} images.")
    return np.array(dataset_images), np.array(dataset_embeddings)




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

    # Check if the dataset folder exists and is not empty
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        return jsonify({"error": "Dataset not found or is empty. Ensure the dataset is uploaded to the correct location."}), 400

    # Compute embeddings for the datasetheroo
    try:
        dataset_images, dataset_embeddings = compute_dataset_embeddings(dataset_path, embedding_model, target_size)
    except Exception as e:
        return jsonify({"error": f"Failed to process dataset: {e}"}), 500

    # Check if an image is uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Preprocess the uploaded image
    try:
        image_file = request.files["image"]
        if not image_file.filename.lower().endswith((".jpeg", ".jpg", ".png")):
            return jsonify({"error": "Unsupported file format. Please upload a JPEG or PNG image."}), 400

        preprocessed_image = preprocess_image(image_file, target_size)
    except Exception as e:
        return jsonify({"error": f"Failed to preprocess uploaded image: {e}"}), 500

    # Compute the embedding for the uploaded image
    try:
        target_embedding = embedding_model.predict(preprocessed_image)
    except Exception as e:
        return jsonify({"error": f"Failed to generate embedding for uploaded image: {e}"}), 500

    # Find the closest matches
    try:
        closest_images = find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images)
    except Exception as e:
        return jsonify({"error": f"Failed to find closest matches: {e}"}), 500

    # Prepare response with matching results
    results = []
    for i, (image_array, distance) in enumerate(closest_images):
        try:
            # Convert the image array to a PIL Image for the response
            image = Image.fromarray((image_array * 255).astype("uint8"))
            img_io = BytesIO()
            image.save(img_io, "PNG")
            img_io.seek(0)

            results.append({"index": i, "distance": float(distance)})
        except Exception as e:
            return jsonify({"error": f"Failed to prepare response for image {i}: {e}"}), 500

    return jsonify({"results": results})


if __name__ == "__main__":
    print("Initializing resources...")
    setup()
    app.run(debug=False)  # Disable debug mode for production
    #sadhnfgia