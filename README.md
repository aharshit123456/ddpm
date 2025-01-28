# DDPM Project

This repository contains the implementation of Denoising Diffusion Probabilistic Models (DDPM).

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction
Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn to generate data by reversing a diffusion process. This repository provides a comprehensive implementation of DDPM.

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage
To train the model, use the following command:
```bash
python train.py
```
To generate samples, use:
```bash
python generate.py
```

## Game
To understand the model and it's workings, we're working on a cool cute little game where the user is the UNET reverser/diffusion model and is tasked to denoise the images with noise made of grids of lines.

## Explanations and Mathematics
- slides from presentation : 
- notes/explanations : [HERE](slides\notes)
- a cute lab talk ppt: 
- plato's allegory : \<link to REPUBLIC>

## Resources
- Original Paper : https://arxiv.org/pdf/2006.11239
- Improvement Paper : https://arxiv.org/abs/2102.09672
- Improvement by OpenAI : https://arxiv.org/pdf/2105.05233
- Stable Diffusion Paper : https://arxiv.org/abs/2112.10752
- 

### Papers for background
- UNET Paper for Biomedical Segmentation
- Autoencooder
- Variational Autoencoder
- Markov Hierarchical VAE
- Introductory Lectures on Diffusion Process

### Youtube videos and courses
#### Mathematics
- Outliers
- Omar Jahil

#### Pytorch Implementation
- [Deep Findr](https://www.youtube.com/watch?v=a4Yfz2FxXiY)
- [Notebook from Deep Findr](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing)

## Pretrained Weights
weights from the model can be found in [pretrained_weights](https://drive.google.com/drive/folders/1NiQDI3e67I9FITVnrzNPP2Az0LABRpic?usp=sharing)

For loading the pretrained weights:
```
model2 = SimpleUnet()
model2.load_state_dict(torch.load("/content/drive/MyDrive/Research Work/mlsa/DDPM/model_weights.pth"))
model2.eval()
```

For making inferences
TODO: Errors in the sampling function, boolean errors and etc. Will open issues for solving by others as exercise if needed.
```
num_samples = 8  # Number of images to generate
image_size = (3, 32, 32)  # Example for CIFAR10
noise = torch.randn(num_samples, *image_size).to("cuda")

model2.to("cuda")
# Generate images by denoising
with torch.no_grad():
    generated_images = model2.sample(noise)

# Save the generated images
save_image(generated_images, "generated_images.png", nrow=4, normalize=True)
```


## Contributing
Contributions are welcome! Please open an issue or submit a pull request.


## Future Ideas
- Make the model onnx compatible for training and inferencing on Intel GPUs
- Build a Stable Diffusion model Text2Img using CLIP implementationnnnn !!!
- Train the current model for a much larger dataset with more generalizations and nuances