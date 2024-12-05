from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from io import BytesIO
from PIL import Image
import numpy as np
import os
import boto3
import tensorflow as tf
import io

app = Flask(__name__)
CORS(app)

MODEL_FILE_KEY = "siamese_model.h5"  # S3 key for the model file
FOLDER_PREFIX = "tables-20241203T011356Z-001/tables/anchor"  # S3 prefix for the folder
FOLDER_LOCAL_DIR = "/tmp/tables"  # Local directory to store the folder

# Load model dynamically from S3
def load_models():
    """Load the TensorFlow model directly from S3."""
    bucket_name = os.getenv("S3_BUCKET_NAME")
    model_key = MODEL_FILE_KEY

    # Create an S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    try:
        # Stream the model file from S3
        print(f"Downloading model {model_key} from bucket {bucket_name}...")
        model_object = s3.get_object(Bucket=bucket_name, Key=model_key)
        model_data = model_object["Body"].read()
        model_bytes = io.BytesIO(model_data)

        # Load the model from the byte stream
        custom_objects = {"Addons>TripletSemiHardLoss": tf.keras.losses.CategoricalCrossentropy}
        siamese_model = tf.keras.models.load_model(model_bytes, custom_objects=custom_objects)

        # Extract the embedding model (adjust index as needed)
        embedding_model = siamese_model.layers[3]
        print(f"Model {model_key} loaded successfully from S3.")
        return siamese_model, embedding_model

    except Exception as e:
        print(f"Error loading model from S3: {e}")
        raise

# Download folder from S3
def download_folder(bucket_name, s3_prefix, local_dir):
    """Download all files from an S3 folder."""
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )
    
    os.makedirs(local_dir, exist_ok=True)  # Ensure the directory exists
    paginator = s3.get_paginator('list_objects_v2')
    
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
            for obj in page.get('Contents', []):
                file_name = os.path.join(local_dir, obj['Key'].split('/')[-1])
                s3.download_file(bucket_name, obj['Key'], file_name)
                print(f"Downloaded {obj['Key']} to {file_name}")
        print(f"All files downloaded from {s3_prefix} to {local_dir}.")
    except Exception as e:
        print(f"Error downloading folder from S3: {e}")
        raise

# Setup function
def setup():
    """Download the folder resources at startup."""
    bucket_name = os.getenv("S3_BUCKET_NAME")
    download_folder(bucket_name, FOLDER_PREFIX, FOLDER_LOCAL_DIR)
    print("Setup complete. Folder resources downloaded.")

@app.route('/')
def index():
    return "Flask Heroku App"

# Preprocess image
def preprocess_image(image, target_size=(28, 28)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Find closest embeddings
def find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images, k=5):
    distances = np.linalg.norm(dataset_embeddings - target_embedding, axis=1)
    closest_indices = np.argsort(distances)[:k]
    return [(dataset_images[idx], distances[idx]) for idx in closest_indices]

# API route to process uploaded image
@app.route('/api/upload', methods=['POST'])
@cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
def upload_image():
    """Handles image uploads, processes them, and returns matching results."""
    # Load the model
    _, embedding_model = load_models()

    # Use the global folder path for dataset
    dataset_path = FOLDER_LOCAL_DIR
    target_size = (28, 28)

    # Load and compute dataset embeddings
    dataset_images = []  # Example placeholder for dataset images
    dataset_embeddings = []  # Example placeholder for embeddings
    # Populate these variables based on your application needs

    # Check if an image is uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Preprocess the uploaded image
    image_file = request.files['image']
    preprocessed_image = preprocess_image(image_file, target_size)

    # Compute the embedding for the uploaded image
    target_embedding = embedding_model.predict(preprocessed_image)

    # Find the closest matches
    closest_images = find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images)

    # Prepare response with matching results
    results = []
    for i, (image_array, distance) in enumerate(closest_images):
        # Convert the image array to a PIL Image for the response
        image = Image.fromarray((image_array * 255).astype('uint8'))
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        # Add the result
        results.append({"index": i, "distance": float(distance)})

    return jsonify({"results": results})

if __name__ == "__main__":
    print("Initializing resources...")
    setup()  # Ensure the folder resources are downloaded
    app.run(debug=False)  # Start the Flask app