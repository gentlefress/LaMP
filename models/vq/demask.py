import torch
import torch.nn as nn
from einops import rearrange

import os, sys
# sys.path.append(os.getcwd())
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
class ResnetBlockwithKernel(nn.Module):
    def __init__(self, *, in_channels, out_channels = None, conv_shortcut = False, dropout = 0., temb_channels = 512, kernel_size = 3):
        super().__init__()
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0
        else:
            raise NotImplementedError()
            
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = padding)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x, temb=None, **ignore_kwargs):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class BiasedSelfAttnBlock(nn.Module):
    def __init__(self, in_channels, reweight = False):
        super().__init__()
        self.in_channels = in_channels
        self.apply_reweight = reweight
        
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, mask, **ignore_kwargs):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,t = q.shape
        q = q.permute(0,2,1)   # b,hw,c
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        
        if mask is not None:
            unsqueezed_mask = mask.unsqueeze(-2)
            w_ = w_ * unsqueezed_mask
            
            if self.apply_reweight:
                w_sum = torch.sum(w_, dim=-1, keepdim=True)
                w_ = w_ / w_sum

        # attend to values
        # v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        # h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class TransformerStyleEncoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = BiasedSelfAttnBlock(dim)
        self.resblock = ResnetBlockwithKernel(in_channels=dim)

    def forward(self, x, mask = None):
        x = self.resblock(x)
        x = self.attn(x, mask)
        return x

class TransformerStyleEncoder(nn.Module):
    def __init__(self, dim, n_layer, mask_init_value = 0.02):
        super().__init__()
        self.n_layer = n_layer
        self.mask_init_value = mask_init_value
        self.blocks = nn.ModuleList()
        for i in range(self.n_layer):
            self.blocks.append(TransformerStyleEncoderBlock(dim))
        self.last_resblock = ResnetBlockwithKernel(in_channels=dim)
    
    def forward(self, x, mask = None):
        if mask is not None:
            mask = mask + self.mask_init_value * (1 - mask)  # replace 0 with self.mask_init_value
        for i in range(self.n_layer):
            x = self.blocks[i](x = x, mask = mask)
            if mask is not None:
                mask = torch.sqrt(mask)
        x = self.last_resblock(x)
        return x

class VanillaDemasker(nn.Module):
    def __init__(self, codebook_dim, output_dim, n_layer, mask_init_value = 0.02):
        super().__init__()
        self.output_dim = output_dim
        self.codebook_dim = codebook_dim
        self.mask_token = nn.Parameter(torch.zeros(1, codebook_dim, 1), requires_grad = True)
        self.mask_token.data.uniform_(-1.0 / codebook_dim, 1.0 / codebook_dim)
        self.post_projection = torch.nn.Conv1d(codebook_dim, output_dim, 1)

        self.transformer = TransformerStyleEncoder(output_dim, n_layer, mask_init_value)
        # keep it here to see whether it should
        self.conv_in = torch.nn.Conv1d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, sampled_quant, remain_quant, sample_index, remain_index = None, mask = None, training=True):
        batch_size = sampled_quant.size(0) 
        if training:
            sample_index = sample_index.cpu().numpy().tolist()
            if remain_index == None:
                remain_index = [[j for j in range(16) if j not in sample_index[i]] for i in range(batch_size)]
            else:
                remain_index = remain_index.cpu().numpy().tolist()
            
            full_embedding = nn.Parameter(torch.zeros((batch_size, self.codebook_dim, 16))).to(sampled_quant.device)

            for i in range(batch_size):
                full_embedding[i, :, sample_index[i]] = sampled_quant[i]
                full_embedding[i, :, remain_index[i]] = self.mask_token
        else:
            full_embedding = sampled_quant
        # full_embedding = rearrange(full_embedding, "B C N -> B C H W", H=self.hw, W=self.hw)
        full_embedding = self.post_projection(full_embedding)
        full_embedding = self.conv_in(full_embedding)  # The conv_in in the decoder is here
        recovered_embedding = self.transformer(full_embedding, mask)
        
        return recovered_embedding