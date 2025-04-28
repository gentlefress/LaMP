# This code is based on https://github.com/Mael-zys/T2M-GPT.git
import torch.nn as nn
from models.vq.encdec import Encoder, Decoder
from models.vq.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset, QuantizerCSS, SoftCVQLayer, SoftVQLayer
import clip
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from models.vq.demask import VanillaDemasker
class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        self.encoder = Encoder(251 if args.dataset_name == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.decoder = Decoder(251 if args.dataset_name == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        print('Loading CLIP...')
        self.clip_version = 'ViT-B/32'
        self.clip_model = self.load_and_freeze_clip(self.clip_version)
        self.norm_feature = nn.LayerNorm(code_dim, elementwise_affine=False)
        self.pre_projection = torch.nn.Linear(code_dim, code_dim, bias=False)
        self.mask = torch.from_numpy(np.zeros(16)).float()
        self.demasker = VanillaDemasker(codebook_dim=code_dim, output_dim=code_dim, n_layer=8, mask_init_value = 0.02)
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        elif args.quantizer == 'css':
            self.quantizer = QuantizerCSS(nb_code, code_dim, args)
        elif args.quantizer == "cvq":
            self.quantizer = SoftCVQLayer(10, code_dim, code_dim)
        elif args.quantizer == "softvq":
            self.quantizer = SoftVQLayer(nb_code, code_dim, code_dim)
    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16
        # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x


    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)
        return code_idx

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
    # def forward(self, x, mode, temperature):
    def forward(self, x, text):
        
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        bs = x_encoder.size(0)
        if text is not None:
            with torch.no_grad():
                text_feature = self.encode_text(text)
            motion_feature = F.normalize(x_encoder.permute(0, 2, 1), dim=-1, p=2)
            text_feature = text_feature.unsqueeze(1).expand(-1, motion_feature.size(1), -1)
            pred_score = F.softmax(F.cosine_similarity(text_feature, motion_feature, dim=-1), dim=-1)
            pred_score_clone = pred_score.clone().detach()
            sort_score, sort_order = pred_score_clone.sort(descending=True,dim=-1)
            sort_topk = sort_order[:, :5]
            sort_topk_remain = sort_order[:, 5:]
            ## flatten for gathering
            motion_feature = self.norm_feature(x_encoder.permute(0, 2, 1))

            ## (only) sampled features multiply with score 
            motion_features_sampled = motion_feature.gather(1, sort_order[...,None].expand(-1, -1, motion_feature.size(-1)))
            motion_features_sampled = rearrange(self.pre_projection(motion_features_sampled), "B N C -> B C N")
            self.mask = self.mask.to(motion_features_sampled.device)
            for i in range(bs):
                if i == 0:
                    mask = self.mask.scatter(-1, sort_topk[i], 1.).unsqueeze(0)
                else:
                    mask_i = self.mask.scatter(-1, sort_topk[i], 1.).unsqueeze(0)
                    mask = torch.cat([mask, mask_i], dim=0)
            squeezed_mask = mask.view(bs, -1)  # [batch_size, length]
            x_quantized, loss, perplexity  = self.quantizer(motion_features_sampled)
            sampled_length = sort_topk.size(1)
            sampled_quant = x_quantized[:, :, :sampled_length]
            remain_quant = x_quantized[:, :, sampled_length:]
            h = self.demasker(sampled_quant, remain_quant, sort_topk, sort_topk_remain, squeezed_mask, training=True)
        else:
            motion_feature = self.norm_feature(x_encoder.permute(0, 2, 1))
            x_quantized, loss, perplexity  = self.quantizer(motion_feature.permute(0, 2, 1))
            h = self.demasker(x_quantized, remain_quant=None, sample_index=None, remain_index=None, mask=None, training=False)
        ## quantization
        ## decoder
        x_decoder = self.decoder(h)
        return x_decoder, loss, perplexity


        # SoftVQ
        # x_quantized, vq_code, loss  = self.quantizer(x_encoder, mode=mode, temperature=temperature)
        # x_quantized = x_quantized.permute(0, 2, 1)
        # x_decoder = self.decoder(x_quantized)
        # return x_decoder, loss, vq_code

        # x_out = self.postprocess(x_decoder)
        # return x_out, loss, perplexity


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        return x_decoder
        # x_out = self.postprocess(x_decoder)
        # return x_out



class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataset_name == 'kit' else 22
        self.vqvae = VQVAE_251(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x, text):
    # def forward(self, x, mode, temperature):

        x_out, loss, perplexity = self.vqvae(x, text)
        # x_out, loss, perplexity = self.vqvae(x, mode, temperature)
        return x_out, loss, perplexity

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
        