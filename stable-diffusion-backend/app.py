import numpy as np

# Stable Diffusion Imports
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image
import base64
from io import BytesIO
import os
import cv2
from matplotlib import pyplot as plt

import boto3
import json

from prompt_gen import PromptGenerator

from image_processing import preprocess_image, postprocess_output, generate_mask

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def encode_img(img):
    if isinstance(img, Image.Image):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
    elif isinstance(img, np.ndarray):
        success, buffer = cv2.imencode('.png', img)
        if not success:
            raise ValueError("Could not encode NumPy array to image")
        img_bytes = buffer.tobytes()
    else:
        raise TypeError("Unsupported image type. Expected PIL.Image.Image or numpy.ndarray.")

    encoded_img = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_img

def query_endpoint(payload, endpoint_name, region='us-east-1'):
    encoded_payload = json.dumps(payload).encode('utf-8')
    client = boto3.client(
        "sagemaker-runtime",
        region_name=region
    )
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json;jpeg',
        Accept='application/json;jpeg',
        Body=encoded_payload
    )
    return response

def image_to_image_payload(input_image, mask, positive_prompt, negative_prompt, num_inference_steps=30, guidance_scale=3):
    encoded_input_image = encode_img(input_image)
    encoded_mask_image = encode_img(mask)

    payload = { 
        "prompt": positive_prompt,
        "image": encoded_input_image, 
        "mask_image": encoded_mask_image, 
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt,
        "batch_size": 2
    }
    return payload

@app.route('/generate', methods=['OPTIONS'])
def handle_options():
    response = jsonify({"message": "Preflight check passed"})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response



@app.route('/generate', methods=['POST'])
def generate_image():
    print("here")
    try:
        # Get the uploaded image and text prompt
        uploaded_image = request.files['image']
        input_prompt = request.form['text']
        
        # Save the uploaded image to a temporary path
        image_path = os.path.join(UPLOAD_FOLDER, uploaded_image.filename)
        uploaded_image.save(image_path)
        
        # Open the uploaded image
        original_image = Image.open(image_path).convert("RGB")

        # Generate positive and negative prompts (mock example)
        positive_prompt = f"Positive: {input_prompt}"
        negative_prompt = f"Negative: do not include blur"

        # Create a mask for the image (mock implementation)
        feathered_mask = generate_mask(original_image, mask_highest=False)  # You should implement this function

        # Prepare payload for SageMaker endpoint
        endpoint_name = 'SD-2-Inpaint-Endpoint-DEMO'
        image_payload = image_to_image_payload(original_image, feathered_mask, positive_prompt, negative_prompt)
        
        # Query the endpoint
        response = query_endpoint(image_payload, endpoint_name)

        # Parse and process the response
        response_dict = json.loads(response['Body'].read())
        generated_images = response_dict['generated_images']
        generated_image = base64.b64decode(generated_images[0])

        # Convert the generated image to Base64
        generated_image_base64 = base64.b64encode(generated_image).decode('utf-8')

        # Return the Base64 image string
        return jsonify({
            "success": True,
            "generated_image": generated_image_base64
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
