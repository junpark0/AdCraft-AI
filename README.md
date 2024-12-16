# AdCraft AI

**AdCraft AI** is a powerful tool designed to simplify and elevate the ad creation process by leveraging cutting-edge machine learning models. With AdCraft AI, users can generate visually stunning ads and intelligent captions with minimal input, making it a go-to solution for marketers, designers, and businesses.

## Features

### Ad Creation
- **Image Generation:** Powered by **Stable Diffusion**, users can create high-quality, visually appealing images.
- **Captioning:** Intelligent caption generation using **LLaMA 3.1** for contextual and compelling marketing copy.
- **Object Detection and Masking:** Utilizes **Ultralytics’ YOLOv11 segmentation** model to detect the focal subject and apply Gaussian feathered masking for precise image processing.
- **Prompt Expansion:** Enhanced by **FooocusAI’s GPT-2 model**, thematic keyword-guided prompt generation ensures ad designs align with user intent.

### Caption Refinement
- **Reddit API Integration:** Extracts contextually relevant user insights from Reddit posts to inform ad captions.
- **Data Vectorization:** Identifies top similar posts using keyword searches for optimal contextualization.
- **Summarization:** Combines **Pegasus-xsum** summaries and **BLIP-generated image descriptions** for well-rounded, targeted marketing captions.

---

## Project Structure

The repository contains two main folders:

### Front-End
- Built with modern web development tools for a seamless and intuitive user experience.
- Manages user interactions and integrates with the backend for real-time updates.

### Back-End
- Implements advanced AI models for ad creation and captioning.

### Key Backend Features:
1. **Image Generation**
   - Generates images based on user prompts, starting images, and Gaussian masks using **SageMaker SD-2 Inpaint**.

2. **Caption Generation**
   - Combines insights from Reddit posts and AI-generated image summaries to produce marketing-ready captions.
   - Steps:
     - **Reddit Data Retrieval:** Extracts top posts based on user-specified keywords.
     - **Summarization:** Uses **Pegasus-xsum** to summarize Reddit posts.
     - **BLIP Summaries:** Generates image-specific summaries.
     - **Caption Synthesis:** Merges Reddit and BLIP insights using **LLaMA 3.1** for a polished caption.

3. **Object Detection**
   - Detects and segments key subjects in uploaded images using **YOLOv11**.
   - Generates Gaussian feathered masks around detected subjects for clean image processing.

4. **Prompt Expansion**
   - Generates detailed, thematic prompts using **GPT-2** to guide image and caption creation.

---

## File Structure

### Back-End Highlights
- `ldm_patched/`: Required modules for FooocusAI’s GPT-2 prompt expansion.
- `expansion.py`: Handles prompt expansion with thematic keywords.
- `image_processing.py`: Generates Gaussian feathered masks and handles optional image resizing.
- `prompt_gen/`: Adds thematic keyword-guided prompt generation.
- `image_generation/`: Interacts with SageMaker SD-2 Inpaint for image generation.
- `caption_generation/`: Processes data from Reddit and BLIP models to generate refined captions.
- pytorch_model.bin file for prompt expansion: https://drive.google.com/drive/folders/1x0Ruf0ApKy1PytJzDwD66kvjnEYfrAVL 

### Front-End
- React-based front-end with components for user interaction and integration with the backend.

---

## How to Use

1. **Input Your Prompt:**
   - Provide a starting prompt to guide the ad creation process.
2. **Upload an Image:**
   - Optionally upload an image for customization.
3. **Generate Ads and Captions:**
   - The application generates a polished ad image and caption tailored to your inputs.

---

Thanks!
