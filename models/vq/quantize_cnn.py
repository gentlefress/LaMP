import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings 
warnings.filterwarnings('ignore')
class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = args.mu
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    # def get_codes_from_indices(self, indices): #indices shape 'b n q' # dequantize

    #     batch, quantize_dim = indices.shape[0], indices.shape[-1]

    #     # because of quantize dropout, one can pass in indices that are coarse
    #     # and the network should be able to reconstruct

    #     if quantize_dim < self.num_quantizers:
    #         indices = F.pad(indices, (0, self.num_quantizers - quantize_dim), value = -1)

    #     # get ready for gathering

    #     codebooks = repeat(self.codebooks, 'q c d -> q b c d', b = batch)
    #     gather_indices = repeat(indices, 'b n q -> q b n d', d = codebooks.shape[-1])

    #     # take care of quantizer dropout

    #     mask = gather_indices == -1.
    #     gather_indices = gather_indices.masked_fill(mask, 0) # have it fetch a dummy code to be masked out later

    #     # print(gather_indices.max(), gather_indices.min())
    #     all_codes = codebooks.gather(2, gather_indices) # gather all codes

    #     # mask out any codes that were dropout-ed

    #     all_codes = all_codes.masked_fill(mask, 0.)
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x
        

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity



class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        
        N, width, T = z.shape
        z = self.preprocess(z)
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity

    def quantize(self, z):

        assert z.shape[-1] == self.e_dim

        # B x V
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):

        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x



class QuantizeReset(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.reset_codebook()
        self.codebook = nn.Parameter(torch.randn(nb_code, code_dim))
        
    def reset_codebook(self):
        self.init = False
        self.code_count = None

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = nn.Parameter(out[:self.nb_code])
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        # Update centres
        self.code_count = code_count  # nb_code
        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()

        self.codebook.data = usage * self.codebook.data + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape
        # Preprocess
        x = self.preprocess(x)
        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)
        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)
        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity

    
class QuantizeEMA(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = code_update
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity

# Out
# Canonical Sphere Space Quantization
class QuantizerCSS(nn.Module): 
    def __init__(self, n_e, e_dim, beta):
        super(QuantizerCSS, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        self.map = nn.Linear(self.e_dim, 3)
    def forward(self, z):        
        N, width, T = z.shape
        z = self.preprocess(z)
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        z_mapped = self.map(z_flattened)
        z_mapped = F.normalize(z_mapped, dim=1, p=2)
        mapped_codebook = self.map(self.embedding.weight)
        mapped_codebook = self.map_to_unit_sphere(mapped_codebook)
        # B x V
        d = self.great_circle_distance_torch(z_mapped, mapped_codebook)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return z_q, loss, perplexity
    def map_to_unit_sphere(self, points):
        norms = torch.norm(points, dim=1, keepdim=True)
        unit_vectors = torch.where(norms != 0, points / norms, torch.zeros_like(points))
        return unit_vectors
    def great_circle_distance_torch(self, point1, point2, radius=1.0):
        # 向量的点积
        dot_product = torch.matmul(point1, point2.t())
        
        # 向量的模长
        magnitude_p1 = torch.norm(point1, p=2)
        magnitude_p2 = torch.norm(point2, p=2)
        
        # 球心角的余弦值
        cos_theta = dot_product / (magnitude_p1 * magnitude_p2)
        
        # 限制在[-1, 1]范围内，防止浮点误差导致的溢出
        cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
        
        # 计算球心角（弧度）
        theta = torch.acos(cos_theta)
        
        # 计算并返回测地距离
        return radius * theta
    def quantize(self, z):

        assert z.shape[-1] == self.e_dim

        # B x V
        d = torch.sum(z ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def dequantize(self, indices):

        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x
class LookupFreeQuantizer(nn.Module):
    def __init__(self, vocab_size: int=None, hidden_size=512):
        super(LookupFreeQuantizer, self).__init__()
        self.proj = nn.Linear(hidden_size, int(math.log2(vocab_size)))
        self.vocab_size = vocab_size
        self.proj_inv = nn.Linear(int(math.log2(vocab_size)), hidden_size)
        self.MSE = nn.L1Loss(reduction='none')

    def sign(self, z: torch.Tensor):
        q_z = torch.sign(z)
        q_z[q_z == 0] = 1
        return q_z

    def token_index(self, q_z: torch.Tensor):
        indices = (torch.arange(q_z.size(-1), dtype=torch.float32)).to(q_z.device)
        tokens = torch.sum(2**indices * (q_z > 0).float(), dim=-1)
        return tokens
    
    def get_vq(self, z: torch.Tensor, attn_mask=None, mode='train', temperature=1.0):
        z = F.tanh(self.proj(z))
        if self.vocab_size is not None:
            assert z.size(-1)==math.log2(self.vocab_size)
        q_z = self.sign(z)
        q_z = z + (q_z - z).detach()
      #  index = self.token_index(q_z)
        index = (q_z > 0).long()
        return index, None

    def embed_id(self, q_z):
        quantized = self.proj_inv(q_z[:,:,:-1].float())
        return quantized

    def forward(self, z: torch.Tensor, attn_mask=None, mode='train', temperature=1.0):
        z = F.tanh(self.proj(z.permute(0, 2, 1)))
        if self.vocab_size is not None:
            assert z.size(-1)==math.log2(self.vocab_size)

        q_z = self.sign(z)
        q_z = z + (q_z - z).detach()
        # if mode == 'train':
        #     q_z = torch.sigmoid(z/temperature)
        # else:
        #     # q_z = self.sign(z)
        #     temperature = 1e-5
        #     q_z = torch.sigmoid(z/temperature)
        vq_loss = self.MSE(z.detach(), q_z) + 0.25*self.MSE(z, q_z.detach())
        if attn_mask is not None:
            vq_loss = ((vq_loss*attn_mask[:,:,None]).sum(dim=1)/attn_mask[:,:,None].sum(dim=1)).mean()
        else:
            vq_loss = vq_loss.mean()
            
        q_z = self.proj_inv(q_z)
        index = self.token_index(q_z)
        return q_z, vq_loss, index.int()
class SoftCVQLayer(nn.Module):
    def __init__(self, log2_num_embeddings, embedding_dim, vq_dim, condition_layer=6, sphere=True):
        super(SoftCVQLayer, self).__init__()
        self.init = True
        self.log2_num_embeddings = log2_num_embeddings
        # 生成从0到65535的整数范围
        int_range = torch.arange(0, 2**log2_num_embeddings)
        bool_vectors = (int_range[:, None] & (1 << torch.arange(log2_num_embeddings-1, -1, -1))) > 0
        self.embedding = nn.Parameter(bool_vectors.float(), requires_grad=False)
        self.sphere = sphere

        '''
        SoftBV-vq16-conditional: 使用两层MLP将bool vector映射到log2_num_embeddingslog2_num_embeddings相同维度, 不归一化
        SoftBV-vq16-conditional-sphere: 使用两层MLP将bool vector映射到log2_num_embeddings相同维度, 归一化
        SoftBV-vq16-conditional-mlp2-vqdim32: 使用两层MLP将bool vector映射到vq_dim, 不归一化
        SoftBV-vq16-conditional-sphere-vqdim32: 使用两层MLP将bool vector映射到vq_dim, 归一化
        SoftBV-vq16-conditional-mlp3-vqdim32: 使用三层MLP将bool vector映射到vq_dim, 不归一化
        '''
        hidden_dim = 1024

        if condition_layer <=3:
            layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.ReLU()]
            for _ in range(condition_layer - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim)),
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, vq_dim))
            self.embedding_mlp = nn.Sequential(*layers)
        else:
            layers = [nn.Linear(log2_num_embeddings, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
            for _ in range(condition_layer - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, vq_dim))
            self.embedding_mlp = nn.Sequential(*layers)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        # self.proj_trans =  nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)
        
        self.init=False
        self.MSE = nn.HuberLoss(reduction='none')
        
    def project(self, h): 
        h = self.proj(h)
        return h
    
    def project_inv(self, h):
        h = self.proj_inv(h)
        return h

    def embed_id(self, vq_id):
        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            embed = embed/torch.norm(embed, dim=-1, keepdim=True) # spherical
        return self.proj_inv(embed[vq_id])

    def attention(self, H, C, temperature=1):
        distances = torch.sum(H ** 2, dim=1, keepdim=True) \
                  + torch.sum(C ** 2, dim=1) \
                  - 2 * torch.matmul(H, C.t())
        
        # distances = torch.cdist(H, C, compute_mode='use_mm_for_euclid_dist')
        A = F.softmax(-distances/(temperature+1e-8), dim=1)
        # A = gumbel_softmax(-distances, temperature)
        # A = F.softmax(-distances, dim=1)
        return A, -distances

    def normalize(self, x):
        return x/(torch.norm(x, dim=-1, keepdim=True)+1e-6)
    
    def get_vq(self,h, attn_mask = None, temperature = 1e-5):
        h = self.proj(h)

        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical
        
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,0])

        h_flat = h[attn_mask==1]
        A = self.attention(h_flat, embed, temperature)
        code = A.argmax(dim=-1)
        h_vq = embed[code]
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code
        quantized = self.proj_inv(quantized)
        return vq_code, quantized
    
    def entropy_loss(self, P, Q):
        return -torch.sum(P * torch.log(Q))


    def forward(self, h_in, attn_mask=None, mode='train', temperature=1):
        # h_in = self.proj_trans(h_in)
        h_in = h_in.permute(0, 2, 1)
        h = self.proj(h_in)
        embed = self.embedding_mlp(self.embedding)
        if self.sphere:
            h = self.normalize(h) # spherical
            embed = self.normalize(embed) # spherical
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:, :, 0])
        h_flat = h[attn_mask==1]
        
        A, flat_affinity = self.attention(h_flat, embed, temperature)
        # entropy = -torch.sum(A * torch.log(A), dim=-1)
        # entropy_neg = -(entropy*attn_mask).sum()/attn_mask.sum()

        code = A.argmax(dim=-1)
        # W_decay = torch.ones_like(A)*temperature
        # idx = torch.arange(code.shape[0], device=code.device)
        # W_decay[idx, code] = 1
        # A = A*W_decay
        # A = A/A.sum(dim=-1, keepdim=True)

        # idx = torch.arange(code.shape[0], device=code.device)
        # W = torch.ones_like(W_model)*(W_model[idx, code][:,None]).detach()
        
        # W[idx, code] = W_model[idx, code]
        # W[]
        if mode=='train':
            h_vq = A@embed
        else:
            h_vq = embed[code]
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_loss = torch.zeros(1, device=h.device)
        
        vq_code = torch.zeros(h.shape[:-1], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code

        quantized = self.proj_inv(quantized)

        if mode == 'train':
            N = h_in.shape[0]
            keep_idx = torch.randperm(N)[:int(1.0*temperature*N)]
            replace = torch.zeros(h_in.shape[0],1,device=h_in.device,dtype=h_in.dtype)
            replace[keep_idx] = 1
            quantized = quantized*(1-replace).unsqueeze(1) + h_in*replace.unsqueeze(1)
        


        # # 计算每个随机变量的平均值
        # probs = F.softmax(flat_affinity, dim=-1)
        # log_probs = F.log_softmax(flat_affinity + 1e-8, dim=-1)
        # mean_probs = torch.mean(A, dim=0)
        # # 计算均匀分布的目标
        # uniform_target = torch.full_like(mean_probs, 1.0 / mean_probs.size(-1))
        
        # # 计算每个随机变量的熵
        # sample_entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # # 计算所有样本的平均熵
        # avg_sample_entropy = torch.mean(sample_entropy)
        
        # # 计算均匀分布的熵
        # uniform_entropy = -torch.sum(uniform_target * torch.log(uniform_target + 1e-8))
        
        # # 计算目标：使得平均熵接近于均匀分布的熵
        # vq_loss = -0.1*(avg_sample_entropy - uniform_entropy)
        vq_loss = 0

        
        return quantized, vq_code, vq_loss

class SoftVQLayer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, vq_dim, ema_k=5, ema_lambda=0.95, ema_weight=100):
        super(SoftVQLayer, self).__init__()
        self.init = True
        self.ema_k = ema_k
        self.ema_lambda = ema_lambda
        self.ema_weight = ema_weight
        self.num_embeddings = num_embeddings
        self.vq_dim = vq_dim
        self.proj = nn.Linear(embedding_dim, vq_dim)
        self.proj_inv = nn.Linear(vq_dim, embedding_dim)
        self.embedding = nn.Embedding(num_embeddings, vq_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.register_buffer('count', torch.zeros(num_embeddings))
        self.init=False
        self.MSE = nn.HuberLoss(reduction='none')
        
    def project(self, h): 
        h = self.proj(h)
        h = h/h.norm(dim=-1, keepdim=True)
        return h
    
    def project_inv(self, h):
        h = self.proj_inv(h)
        return h

    def embed_id(self, vq_id):
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        return self.proj_inv(self.embedding(vq_id))

    def attention(self, H, C, temperature=1):
        distances = torch.sum(H ** 2, dim=1, keepdim=True) \
                  + torch.sum(C ** 2, dim=1) \
                  - 2 * torch.matmul(H, C.t())
        A = F.softmax(-distances/temperature, dim=1)
        return A
    
    def get_vq(self,h,attn_mask,temperature):
        h = self.proj(h)
        h = h/torch.norm(h, dim=-1, keepdim=True)
        h_flat = h[attn_mask==1]
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)
        A = self.attention(h_flat, self.embedding.weight, temperature)
        code = A.argmax(dim=-1)
        h_vq = self.embedding(code)
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code
        quantized = self.proj_inv(quantized)
        return vq_code, quantized
        
    
    def forward(self, h, attn_mask=None, mode='train', temperature=1):
        h = h.permute(0, 2, 1)
        h = self.proj(h)
        h = h/torch.norm(h, dim=-1, keepdim=True)
        if attn_mask is None:
            attn_mask = torch.ones_like(h[:,:,0])
        h_flat = h[attn_mask==1]
        
        self.embedding.weight.data = self.embedding.weight.data/self.embedding.weight.data.norm(dim=-1, keepdim=True)

        
        A = self.attention(h_flat, self.embedding.weight, temperature)
        code = A.argmax(dim=-1)
        
        if mode=='train':
            h_vq = A@self.embedding.weight
        else:
            h_vq = self.embedding(code)
        
        quantized = torch.zeros_like(h)
        quantized[attn_mask==1] = h_vq
        vq_loss = 0
        
        vq_code = torch.zeros(h.shape[:2], device = h.device, dtype = torch.long)
        vq_code[attn_mask==1] = code

        quantized = self.proj_inv(quantized)
        return quantized, vq_code, vq_loss