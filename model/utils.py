## Scheduler
'''
sequentially add noise to images
precomputed values used
'''

import torch.nn.functional as F
import torch
from precomputes import betas, sqrt_recip_alphas, sqrt_alphas_cumulative_products, sqrt_one_minus_alphas_cumulative_products, posterior_variance
# from model import model
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

def get_index_from_list(vals, t, x_shape):
  batch_size = t.shape[0]
  out = vals.gather(-1, t.cpu())
  return out.reshape(batch_size, *((1,)* (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
  noise = torch.randn_like(x_0)
  sqrt_alphas_cumulative_products_t = get_index_from_list(sqrt_alphas_cumulative_products, t, x_0.shape)
  sqrt_one_minus_alphas_cumulative_products_t = get_index_from_list(
      sqrt_one_minus_alphas_cumulative_products, t, x_0.shape
  )
  ## formulae for image augged looks like sqrt(pi(alpha_t)) * x_t-1 * sqrt(pi(1-alpha_t)) * noise~N(0,1)
  return sqrt_alphas_cumulative_products_t.to(device) * x_0.to(device) \
      + sqrt_one_minus_alphas_cumulative_products_t.to(device) * noise.to(device), noise.to(device)


@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumulative_products_t = get_index_from_list(
        sqrt_one_minus_alphas_cumulative_products, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumulative_products_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image(model, IMG_SIZE=64, device="cpu", T=300):
    
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(model, img.detach().cpu())
    # plt.show()
    return img

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))