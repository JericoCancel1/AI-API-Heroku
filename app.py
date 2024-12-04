from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from io import BytesIO
from PIL import Image
import numpy as np
import os
import boto3

def setup():
    """Download necessary files when the app starts."""
    bucket_name = os.getenv("S3_BUCKET_NAME")
    model_file_key = "siamese_model.h5"  # S3 key for the model file
    model_local_path = "/tmp/siamese_model.h5"  # Local path to store the model
    
    folder_prefix = "tables-20241203T011356Z-001/"  # S3 prefix for the folder
    folder_local_dir = "/tmp/tables"  # Local directory to store the folder
    
    # Download the model file
    download_file(bucket_name, model_file_key, model_local_path)
    
    # Download the folder
    download_folder(bucket_name, folder_prefix, folder_local_dir)

    print("Setup complete. Model and tables downloaded.")

def download_file(bucket_name, s3_key, local_path):
    """Download a file from S3."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    
    # Create the parent directories if they don't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Download the file from S3
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"Downloaded {s3_key} to {local_path}")

def download_folder(bucket_name, s3_prefix, local_dir):
    """Download all files from an S3 folder."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    
    os.makedirs(local_dir, exist_ok=True)  # Create the local folder if it doesn't exist
    paginator = s3.get_paginator('list_objects_v2')
    
    # Iterate through the pages of objects in the S3 bucket
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix, Delimiter='/'):
        for obj in page.get('Contents', []):
            file_name = os.path.join(local_dir, obj['Key'].split('/')[-1])
            download_file(bucket_name, obj['Key'], file_name)
    
    print(f"Downloaded all files from {s3_prefix} to {local_dir}")

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Perform setup when the app is created (instead of before_first_request)
    with app.app_context():
        setup()

    # Define the routes
    @app.route('/')
    def index():
        return "Flask Heroku App"

    @app.route('/api/upload', methods=['POST'])
    @cross_origin(origin='*', headers=['Content-Type', 'Authorization'])
    def upload_image():
        tf, _ = load_dependencies()
        _, embedding_model = load_models()

        # Dataset path and target size
        dataset_path = os.getenv("DATASET_PATH", "default_dataset_path")  # Use env variable
        target_size = (28, 28)

        # Load the dataset and compute embeddings
        dataset_images, dataset_embeddings = prepare_dataset(embedding_model, dataset_path, target_size)

        # Check if an image is uploaded
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        # Preprocess and find matches
        image_file = request.files['image']
        preprocessed_image = preprocess_image(image_file, target_size)
        target_embedding = embedding_model.predict(preprocessed_image)
        closest_images = find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images)

        # Prepare response
        results = []
        for i, (image_array, distance) in enumerate(closest_images):
            # Convert the image array to a PIL Image for response
            image = Image.fromarray((image_array * 255).astype('uint8'))
            img_io = BytesIO()
            image.save(img_io, 'PNG')
            img_io.seek(0)
            results.append({"index": i, "distance": float(distance)})

        return jsonify({"results": results})

    # Helper functions (same as your original app)
    def load_dependencies():
        import tensorflow as tf
        import tensorflow_addons as tfa
        return tf, tfa

    def load_models():
        tf, tfa = load_dependencies()
        custom_objects = {"Addons>TripletSemiHardLoss": tfa.losses.TripletSemiHardLoss}
        siamese_model = tf.keras.models.load_model("siamese_model.h5", custom_objects=custom_objects)
        embedding_model = siamese_model.layers[3]  # Update this index as per your model
        return siamese_model, embedding_model

    def preprocess_image(image, target_size=(28, 28)):
        img = Image.open(image)
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    def find_closest_embeddings(target_embedding, dataset_embeddings, dataset_images, k=5):
        distances = np.linalg.norm(dataset_embeddings - target_embedding, axis=1)
        closest_indices = np.argsort(distances)[:k]
        return [(dataset_images[idx], distances[idx]) for idx in closest_indices]

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)  # Enable debug mode for local testing
