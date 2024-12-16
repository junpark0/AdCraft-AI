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

def encode_img(img):
    """
    Converts a PIL Image object to a base64-encoded string.
    """
    # buffered = BytesIO()
    # img.save(buffered, format="PNG")  # Adjust format as needed
    # encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # return encoded_img
    if isinstance(img, Image.Image):
        # If the image is a PIL.Image object
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
    elif isinstance(img, np.ndarray):
        # If the image is a NumPy array
        success, buffer = cv2.imencode('.png', img)
        if not success:
            raise ValueError("Could not encode NumPy array to image")
        img_bytes = buffer.tobytes()
    else:
        raise TypeError("Unsupported image type. Expected PIL.Image.Image or numpy.ndarray.")

    encoded_img = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_img

def query_endpoint(payload, endpoint_name, region='us-east-1'):
    """query the endpoint with the json payload encoded in utf-8 format."""
    #Setting AWS Credentials
    encoded_payload = json.dumps(payload).encode('utf-8')
    client = boto3.client(
        "sagemaker-runtime",
        region_name="us-east-1"
    )
    # Accept = 'application/json;jpeg' returns the jpeg image as bytes encoded by base64.b64 encoding.
    # To receive raw image with rgb value set Accept = 'application/json'
    # To send raw image, you can set content_type = 'application/json' and encoded_image as np.array(PIL.Image.open('low_res_image.jpg')).tolist()
    # Note that sending or receiving payload with raw/rgb values may hit default limits for the input payload and the response size.
    response = client.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json;jpeg', 
                                      Accept = 'application/json;jpeg', 
                                      Body=encoded_payload)
    return response

def image_to_image_payload(input_image, mask, positive_prompt, negative_prompt, num_inference_steps=30, guidance_scale=3):
    # Read the original image
    # original_image = image_path.convert("RGB") #Assuming that the PIL.Image object is the input. (Change to Image.PIL object if you have to)
    # original_image = Image.open("resized_image.png").convert("RGB")

    # Convert the edge map and mask to PIL Images
    encoded_input_image = encode_img(input_image)
    encoded_mask_image = encode_img(mask)

    # Run the pipeline
    payload = { 
        "prompt":positive_prompt,
        "image":encoded_input_image, 
        "mask_image":encoded_mask_image, 
        "num_inference_steps":num_inference_steps,
        "guidance_scale":guidance_scale,
        "negative_prompt": negative_prompt,
        "batch_size":2
    }

    return payload

def display_image(img, title):
    plt.figure(figsize=(12,12))
    plt.imshow(np.array(img))
    plt.axis('off')
    plt.title(title)
    plt.show()

def parse_and_display_response(query_response):
    """Parse the endpoint response and display the generated images"""
    
    response_dict = json.loads(query_response['Body'].read())
    generated_images = response_dict['generated_images']
    
    for generated_image in generated_images:
        with BytesIO(base64.b64decode(generated_image.encode())) as generated_image_decoded:
            with Image.open(generated_image_decoded) as generated_image_np:
                generated_image_rgb = generated_image_np.convert("RGB")
                generated_image_rgb.save(r"C:\Users\kenis\OneDrive\Desktop\Prompt Expansion Testing\outputs\temp.png")
                display_image(generated_image_rgb, "Inpainted Image")
                

if __name__ == '__main__':
    themes = {
        'fantasy': False,
        'sci-fi': False,
        'portrait': False,
        'landscape': True,
        'advertising': False,
        'semi-realistic': True
    }
    
    IMAGE_PATH = r"Input_Photos\yeti_ad_2.jpg"
    #Path for Image in s3 @Ethan
    
    #SD-2-Inpaint Endpoint name (Must change after making anew endpoint, this happens after every delete and creation)
    endpoint_name = 'SD-2-Inpaint-Endpoint-DEMO'
    
    Input_Prompt = "A cup in front of a river"

    # Create an instance of PromptGenerator
    prompt_gen = PromptGenerator(themes, Input_Prompt)
    
    positive_prompt = prompt_gen.get_positive_prompt()
    negative_prompt = prompt_gen.get_negative_prompt()

    # original_image, resized_image = preprocess_image(IMAGE_PATH)
    original_image = Image.open(IMAGE_PATH).convert("RGB")
    
    feathered_mask = generate_mask(original_image)
    
    image_payload = image_to_image_payload(original_image, feathered_mask, positive_prompt, negative_prompt)
    
    response = query_endpoint(image_payload, endpoint_name, region='us-east-1')
    
    # output = postprocess_output(output_image, original_image)
    parse_and_display_response(response)
    
    # output.save("C:/Users/kenis/OneDrive/Desktop/Prompt Expansion Testing/outputs/temp.png")