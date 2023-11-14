import keras
import tensorflow
import numpy
import json
import boto3
import pickle
import os
import io
import cv2
import base64
from PIL import Image
s3_client = boto3.client('s3')

def load_model_from_s3(bucket_name, model_key):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
        model_data = response['Body'].read()

        # Deserialize the model (adjust this based on how your model is saved)
        model = pickle.loads(model_data)

        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        return None

def lambda_handler(event, context):
    bucket_name = os.environ.get('BUCKET_NAME', 'detect-glasses-model')
    model_key = os.environ.get('MODEL_KEY', 'MNv3_model.pkl')
    model = load_model_from_s3(bucket_name, model_key)
    if 'body' in event:
        input_data = get_input_data(event)  # Define your function to extract input data
        result = model.predict(input_data[None])

        response = {
            'statusCode': 200,
            'body': json.dumps({'prediction': result.tolist()})
        }

        return response
    else:
        # Handle the case where 'body' is not present in the event
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No image data in the request body'})
        }

def get_input_data(event):
    if 'body' in event:
        image_bytes = event['body'].encode('utf-8')
        img_b64dec = base64.b64decode(image_bytes)
        img_byteIO = io.BytesIO(img_b64dec)
        image = Image.open(img_byteIO)
        image = numpy.array(image)
        image = cv2.resize(image, (224,224))
        return image
    return None