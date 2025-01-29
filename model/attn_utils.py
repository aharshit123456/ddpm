import torch
from torch import nn
import math
import torch.nn.functional as F


class SelfAttention(nn.Module):
  def __init__(self, chs, num_heads=1, ffn_expansion=4, dropout=0.1):
    super().__init__()
    self.norm = nn.LayerNorm(chs)
    self.attn = nn.MultiheadAttention(embed_dim=chs, num_heads=num_heads, batch_first=True)

    self.ffn = nn.Sequential(
        nn.Linear(chs, chs*ffn_expansion),
        nn.GELU(),
        nn.Linear(chs*ffn_expansion, chs)
    )

    self.norm2 = nn.LayerNorm(chs)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    b,c,h,w = x.shape
    x_reshaped = x.view(b,c,h*w).transpose(1,2)
    
    attn_out, _ = self.attn(self.norm(x_reshaped), self.norm(x_reshaped), self.norm(x_reshaped))
    x_attn = x_reshaped + self.dropout(attn_out)

    ffn_out = self.ffn(self.norm2(x_attn))
    x_out = x_attn + self.dropout(ffn_out)


    x_out = x_out.transpose(1,2).view(b,c,h,w)
    return x_out
  



class CBAM(nn.Module):
  def __init__(self,chs, reduction=16):
    super().__init__()

    self.channel_attn = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(chs, chs//reduction, 1),
        nn.ReLU(),
        nn.Conv2d(chs//reduction, chs, 1),
        nn.Sigmoid()
    )

    self.spatial_attn = nn.Sequential(
        nn.Conv2d(2,1,kernel_size=7,padding=3),
        nn.Sigmoid()
    )

  def forward(self,x):
    ch_wt = self.channel_attn(x)
    x = x*ch_wt

    avg_pool = torch.mean(x, dim=1, keepdim=True)
    max_pool, _ = torch.max(x, dim=1, keepdim=True)
    sp_wt = self.spatial_attn(torch.cat([avg_pool, max_pool], dim=1))
    x = x* sp_wt

    return x




class Block_CBAM(nn.Module):
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

    self.cbam = CBAM(out_ch)

  def forward(self, x, t, ):
    h = self.batch_norm1(self.relu(self.conv1(x)))
    time_emb = self.relu(self.time_mlp(t))
    time_emb = time_emb[(..., ) + (None, ) * 2]
    h = h + time_emb
    h = self.batch_norm2(self.relu(self.conv2(h)))
    
    h = self.cbam(h)
    return self.transform(h)