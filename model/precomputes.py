import torch
from torch.nn import functional as F

T = 300 ## according to the paper

### SOO MMANNYY PRECOMPUTEDD VALUESS TO TRACKKKK
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1. - betas
alphas_cumulative_products = torch.cumprod(alphas, axis=0)
alphas_cumulative_products_prev = F.pad(alphas_cumulative_products[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumulative_products = torch.sqrt(alphas_cumulative_products)
sqrt_one_minus_alphas_cumulative_products = torch.sqrt(1. - alphas_cumulative_products)
posterior_variance = betas * (1. - alphas_cumulative_products_prev) / (1. - alphas_cumulative_products)