'''
THis gile is to contain the DDPM implementation modularized for loading, prediciton and training.
'''

from torch import nn
import math
import torch
from utils import forward_diffusion_sample, sample_timestep, sample_plot_image
import torch.nn.functional as F


class Block(nn.Module):
  def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
    super().__init__()
    self.time_mlp = nn.Linear(time_emb_dim, out_ch)
    if up:
      ## up channel - go big big big bigg from smol smol smol with 3x3 kernel
      self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
      self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
    else:
      self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
      self.transform = nn.Conv2d(out_ch, out_ch, 4,2,1)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.relu = nn.ReLU()
    self.batch_norm1 = nn.BatchNorm2d(out_ch)
    self.batch_norm2 = nn.BatchNorm2d(out_ch)

  def forward(self, x, t, ):
    h = self.batch_norm1(self.relu(self.conv1(x)))
    time_emb = self.relu(self.time_mlp(t))
    time_emb = time_emb[(..., ) + (None, ) * 2]
    h = h + time_emb
    h = self.batch_norm2(self.relu(self.conv2(h)))
    return self.transform(h)
  
class PositionEmbeddings(nn.Module):
  def __init__(self,dim):
    super().__init__()
    self.dim = dim

  def forward(self, time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings
  


class SimpleUnet(nn.Module):
  def __init__(self):
    super().__init__()
    image_channels = 3
    down_channels = (64, 128, 256, 512, 1024)
    up_channels = (1024, 512, 256, 128, 64)
    self.device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dim = 3
    time_emb_dim = 32

    ## timestep stored as positional encoding in terms of sine
    self.time_mlp = nn.Sequential(
        PositionEmbeddings(time_emb_dim),
        nn.Linear(time_emb_dim, time_emb_dim),
        nn.ReLU()
    )

    
    self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
    self.down_blocks = nn.ModuleList([
        Block(down_channels[i], down_channels[i+1], time_emb_dim)
        for i in range(len(down_channels)-1)
    ])
    self.up_blocks = nn.ModuleList([
        Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True)
        for i in range(len(up_channels)-1)
    ])

    ## readout layer
    self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

  def forward(self, x, timestep):
    t = self.time_mlp(timestep)
    x = self.conv0(x)
    residual_inputs = []
    for down in self.down_blocks:
      x = down(x, t)
      residual_inputs.append(x)
    for up in self.up_blocks:
      residual_x = residual_inputs.pop()
      x = torch.cat((x, residual_x), dim=1)
      x = up(x, t)
    return self.output(x)
  
  @torch.no_grad()
  def sample(self, noise):
      """
      Generate an image by denoising a given noise tensor using the reverse diffusion process.

      Args:
          noise (torch.Tensor): Initial noise tensor (e.g., sampled from a Gaussian distribution).
      
      Returns:
          torch.Tensor: Denoised image.
      """
      img = noise  # Start with the provided noise tensor
      T = self.num_timesteps  # Total timesteps for diffusion
      stepsize = 1  # You can adjust if needed

      # Iterate through the timesteps in reverse order
      for i in range(0, T)[::-1]:
          t = torch.full((noise.size(0),), i, device=noise.device, dtype=torch.long)  # Current timestep
          img = sample_timestep(self, img, t)  # Perform one reverse diffusion step
          img = torch.clamp(img, -1.0, 1.0)  # Clamp the image to ensure values stay in [-1, 1]

      return img

  def get_loss(self, x_0, t):
      x_noisy, noise = forward_diffusion_sample(x_0, t, self.device)
      noise_pred = self(x_noisy, t)
      return F.l1_loss(noise, noise_pred)

  def train(self, dataloader, BATCH_SIZE=64,T=300, EPOCHS=50, verbose=True):
      from torch.optim import Adam

      device = "cuda" if torch.cuda.is_available() else "cpu"
      self.to(device)
      optimizer = Adam(self.parameters(), lr=0.001)
      epochs = EPOCHS

      for epoch in range(epochs):
          for step, batch in enumerate(dataloader):
              optimizer.zero_grad()

              t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
              loss = self.get_loss(self, batch[0], t)
              loss.backward()
              optimizer.step()

              if verbose:
                if epoch % 5 == 0 and step % 150 == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                    sample_plot_image(self)

  def test():
    ## TODO: add the testing loop here
    pass