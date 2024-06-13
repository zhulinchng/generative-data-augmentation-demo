---
title: Generative Data Augmentation
emoji: 🖼
colorFrom: purple
colorTo: red
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
---

# Generative Data Augmentation Demo

Main GitHub Repo: [Generative Data Augmentation](https://github.com/zhulinchng/generative-data-augmentation) | Image Classification Demo: [Generative Augmented Classifiers](https://huggingface.co/spaces/czl/generative-augmented-classifiers).

This demo is created as part of the 'Investigating the Effectiveness of Generative Diffusion Models in Synthesizing Images for Data Augmentation in Image Classification' dissertation.

The user can augment an image by interpolating between two prompts, and specify the number of interpolation steps and the specific step to generate the image.

## Demo Usage Instructions

1. Upload an image.
2. Enter the two prompts to interpolate between, the first prompt should contain the desired class of the augmented image, the second prompt should contain the undesired class (i.e., confusing class).

## Configuration

- Total Interpolation Steps: The number of steps to interpolate between the two prompts.
- Interpolation Step: The specific step to generate the image.
- Example for 10 steps:

    ```python
        Total:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        Sampled:          4
    ```

- Seed: Seed value for reproducibility.
- Negative Prompt: Prompt to guide the model away from generating the image.
- Width, Height: The dimensions of the generated image.
- Guidance Scale: The scale of the guide the model on how closely to follow the prompts.

## Metadata

[SSIM Score](https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html): Structural Similarity Index (SSIM) score between the original and generated image, ranges from 0 to 1.
[CLIP Score](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html): CLIP similarity score between the original and generated image, ranges from 0 to 100.

## Local Setup

```bash
git clone https://huggingface.co/spaces/czl/generative-data-augmentation-demo
cd generative-data-augmentation-demo
# Setup the data directory structure as shown above
conda create --name $env_name python=3.11.* # Replace $env_name with your environment name
conda activate $env_name
# Visit PyTorch website https://pytorch.org/get-started/previous-versions/#v212 for PyTorch installation instructions.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url # Obtain the correct URL from the PyTorch website
pip install -r requirements.txt
python app.py
```