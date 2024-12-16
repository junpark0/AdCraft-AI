#Imports
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from ultralytics import YOLO

# # Import common Detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

def preprocess_image(image_path, target_size=(512, 512)):
    """Taking an input image and resizing it to the require 512x512 format for the SD-inpaint-2 model

    Args:
        image_path (String): Image File Path (Inputted locally or in S3, up to Ethan)
        target_size (tuple, optional): target size for image. Defaults to (512, 512).

    Returns:
        PIL.Image, PIL.Image: Returns the input image in PIL.Image format and the resized 512x512 image in PIL.Image format
    """
    # Load the original image
    original_image = Image.open(image_path).convert("RGB")

    # Resize the image to the target size
    resized_image = original_image.resize(target_size, Image.LANCZOS)

    return original_image, resized_image

def postprocess_output(output_image, original_image):
    """Returns 512x512 output from SD-2-inpaint model to the original resoluation of the input image

    Args:
        output_image (_type_): _description_
        original_image (_type_): _description_

    Returns:
        PIL.Image: Returns the inpaint model output in the same resolution as the original input image
    """
    # Resize the output image to match the original image size
    target_size = original_image.size
    output_resized = output_image.resize(target_size, Image.LANCZOS)
    return output_resized

def generate_mask(input_image, blur_size=21, mask_highest=True):
    model = YOLO("yolo11n-seg.pt")
    
    image = np.array(input_image)
    
    results = model(image, show=False)
    
    for result in results:
        masks = result.masks.data  # Binary masks (N, H, W), where N is the number of objects
        scores = result.boxes.conf  # Confidence scores (N,)

    # Get image dimensions
    image_height, image_width, _ = image.shape

    # Convert scores to a NumPy array
    scores_np = scores.cpu().numpy()  # Convert PyTorch tensor to NumPy array

    # Find the index of the highest score
    max_score_index = np.argmax(scores_np)  # Index of the object with the highest score

    # Get the corresponding mask and score
    highest_mask = masks[max_score_index].cpu().numpy()  # Mask for the highest scoring object
    
    if mask_highest:
        # Find the index of the highest score
        max_score_index = np.argmax(scores_np)  # Index of the object with the highest score

        # Get the corresponding mask
        highest_mask = masks[max_score_index].cpu().numpy()  # Mask for the highest scoring object

        # Resize the mask to match the image dimensions
        combined_mask = cv2.resize(highest_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        combined_mask = (combined_mask * 255).astype(np.uint8)
    else:
        # Initialize an empty mask for all objects
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        for i in range(len(scores_np)):
            # Resize each mask to match the image dimensions
            object_mask = masks[i].cpu().numpy()
            resized_mask = cv2.resize(object_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
            # Add the resized mask to the combined mask
            combined_mask = np.maximum(combined_mask, (resized_mask * 255).astype(np.uint8))
    
    # Convert the mask to binary (0 and 255)
    binary_mask = cv2.bitwise_not(combined_mask)

    # Detect edges using Sobel or another gradient-based method
    edges = cv2.Canny(binary_mask, 50, 150)  # Fine-tune thresholds for better edge detection

    # Dilate the edges slightly to create a thicker boundary (optional)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Create a mask for the blurred region
    blurred_edges = cv2.GaussianBlur(dilated_edges, (blur_size, blur_size), 0)

    # Combine the original binary mask with the blurred regions
    combined_mask = np.where(dilated_edges > 0, blurred_edges, binary_mask).astype(np.uint8)

    return cv2.cvtColor(combined_mask, cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    IMAGE_PATH = r'Input_Photos\yeti_ad_2.jpg'
    
    original_image = Image.open(IMAGE_PATH).convert("RGB")
    
    result = generate_mask(original_image, mask_highest=False)
    
    plt.imshow(result)
    plt.axis('off')
    plt.show()

    # # Optional: Save the black-and-white mask image
    output_path = r"C:\Users\kenis\OneDrive\Desktop\Prompt Expansion Testing\outputs\highest_score_mask_output.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    # print(f"Mask of the highest confidence object saved to {output_path}")