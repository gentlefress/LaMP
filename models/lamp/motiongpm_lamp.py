"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import math
from models.lamp.basemodel import all_gather_with_grad, concat_all_gather
from models.lamp.QFormer_Base import (
    QFormer_Base,
    disabled_train,
)
from utils.metrics import *
import clip
from models.lamp.QFormer_output import QFormer_Output, QFormer_OutputFeatures
import warnings
warnings.filterwarnings("ignore")
class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x
class PositionalEncoding(nn.Module):
    #Borrow from MDM, the same as above, but add dropout, exponential may improve precision
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) #[max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
class LaMP(QFormer_Base):
    """
    GPT4Point first-stage model with Q-former and Point Encoder.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/gpt4point/gpt4point_stage1.yaml",
    }

    def __init__(
        self,
        opt,
        vq_model,
        motion_model="motion_encoder",
        freeze_motion_encoder=True,
        num_query_token=34,
        # 32 + 2 mean and std
        # num_query_token=32,
        cross_attention_freq=2,
        embed_dim=512,
        max_txt_len=32,
        ckpt_special_strs=None,
    ):
        super().__init__()
        self.code_dim = 512
        self.num_tokens = 512
        self.latent_dim = 256
        self.dropout = 0.1
        self.ckpt_special_strs = ckpt_special_strs
        self.vq_model = vq_model
        '''Motion Qformer'''
        self.Qformer, self.query_tokens = self.init_Qformer(                    
            num_query_token, 1408, cross_attention_freq         # self.point_encoder.num_features: 1408
        )
        '''Bert tokenizer'''
        self.tokenizer = self.init_tokenizer() # Bert tokenizer
        '''motion encoder'''
        self.motion_encoder = QFormer_Base.init_motion_encoder(motion_model, opt, opt.dataset_name)
        '''Motion Qformer'''
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        '''Text and others'''
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)    # not freeze
        self.motion_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)   # not freeze                  
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)             # actually it is ptm head
        self.motion_projection = nn.Parameter(torch.empty(512, 1408))
        nn.init.normal_(self.motion_projection, std=1408 ** -0.5)
        self.text_projection = nn.Parameter(torch.empty(self.Qformer.config.hidden_size, 1408))   
        nn.init.normal_(self.text_projection, std=1408 ** -0.5)
        self.motion_cls = nn.Linear(self.Qformer.config.hidden_size, self.num_tokens, bias=False)
        self.temp = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # not freeze
        self.max_txt_len = max_txt_len
        for name, param in self.vq_model.named_parameters():             # PIT frozen 1
            param.requires_grad = False
        for name, param in self.motion_encoder.named_parameters():             # PIT frozen 1
            param.requires_grad = False
        logging.info("freeze VQ")
        # if freeze_motion_encoder:
        #     for name, param in self.motion_encoder.named_parameters():             # PIT frozen 1
        #         param.requires_grad = False
        #     logging.info("freeze motion encoder")

    def __init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    def _prepare_motion_encoder(self):
        vq_ckpt = self.vq_model.state_dict()
        base_ckpt = {k.replace("vqvae.encoder.", ""): v for k,
                         v in vq_ckpt.items()}
        self.motion_encoder.load_state_dict(base_ckpt, strict=False)
    def forward(self, motion, text):
        '''`Inp`ut: motion & text'''
        '''Motion Encoder and Motion Q-Former'''
        motion = motion.permute(0, 2, 1)
        motion_embeds = self.motion_encoder(motion).permute(0, 2, 1)
 
        motion_atts = torch.ones(motion_embeds.size()[:-1], dtype=torch.long).to(             # [bs, 49]
            motion.device
        )
        motion_embeds = motion_embeds @ self.motion_projection
        query_tokens = self.query_tokens.expand(motion_embeds.shape[0], -1, -1)              # [1, 32, 768] -> [bs, 32, 768]
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,              # [bs, 49, 768]
            encoder_hidden_states=motion_embeds,     # [bs, 49, 1408]
            encoder_attention_mask=motion_atts,      # [bs, 49]
            use_cache=True,
            return_dict=True,
        )
        '''motion feature'''
        motion_feats = F.normalize(
            self.motion_proj(query_output.last_hidden_state), dim=-1
        )
        motion_re_feats = F.normalize(                                                             # [bs, 256]
            torch.mean(motion_feats, dim=1), dim=-1
        )
        '''Text Bert'''
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(motion.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(                                                             # [bs, 256]
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )
        ###============== Motion-text Contrastive ===================###
        motion_feats_all = concat_all_gather(                                                # [bs, 32, 256] -> [bs, 32, 256]
            motion_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]     # [bs, 256] -> [bs, 256]

        sim_q2t = torch.matmul(                                                             # [bs, bs, 32]
            motion_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]
        # point-text similarity: aggregate across all query tokens
        sim_p2t = self.temp * sim_q2t
        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), motion_feats_all.permute(0, 2, 1)
        ).squeeze()
        sim_t2p = self.temp * sim_t2q# [batch_size, batch_size*num_gpu]
        # rank = dist.get_rank()
        bs = motion.size(0)
        targets = torch.linspace(0, bs - 1, bs, dtype=int).to(
            motion.device
        )
        loss_p2t = F.cross_entropy(sim_p2t.max(-1)[0], targets, label_smoothing=0.1)
        loss_t2p = F.cross_entropy(sim_t2p.max(-1)[0], targets, label_smoothing=0.1)
        loss_ptc = (loss_p2t + loss_t2p) / 2
        ###============== Motion-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)                 # [bs, 32]
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)       # [bs, 32]
        motion_embeds_world = all_gather_with_grad(motion_embeds)                         # [bs, 257, 1408]
        with torch.no_grad():  
            sim_t2p.max(-1)[0][:, 0 : bs].fill_diagonal_(-10000)
            sim_p2t.max(-1)[0][:, 0 : bs].fill_diagonal_(-10000)            
            
            weights_t2p = F.softmax(sim_t2p.max(-1)[0], dim=1)
            weights_p2t = F.softmax(sim_p2t.max(-1)[0], dim=1)
        # select a negative point for each text
        motion_embeds_neg = []
        
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2p[b], 1).item()
            motion_embeds_neg.append(motion_embeds_world[neg_idx])
        motion_embeds_neg = torch.stack(motion_embeds_neg, dim=0)

        # select a negative text for each point
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_p2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0             # [bs, 32], [bs, 32], [bs, 32] = [3bs, 32]
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],        # [3*bs, 32]
            dim=0,
        )

        
        query_tokens_ptm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)          # [3*bs, 32, 768]
        query_atts_ptm = torch.ones(query_tokens_ptm.size()[:-1], dtype=torch.long).to(     # [3*bs, 32]
            motion.device
        )
        attention_mask_all = torch.cat([query_atts_ptm, text_atts_all], dim=1)              # [3*bs, 32*2]

        motion_embeds_all = torch.cat(                                                       # [3*bs, 257, 1408]
            [motion_embeds, motion_embeds_neg, motion_embeds], dim=0
        )  # pos, neg, pos
        motion_atts_all = torch.ones(motion_embeds_all.size()[:-1], dtype=torch.long).to(     # [3*bs, 257]
            motion.device
        )

        output_ptm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_ptm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=motion_embeds_all,
            encoder_attention_mask=motion_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_ptm.last_hidden_state[:, : query_tokens_ptm.size(1), :]      # [3*bs, 32, 768]
        vl_output = self.itm_head(vl_embeddings)                                            # [3*bs, 32, 768] -> [3*bs, 32, 2]
        logits = vl_output.mean(dim=1)                                                      # [3*bs, 32, 2] -> [300, 2]s

        ptm_labels = torch.cat(                                                             # [3*bs] bs:1, 2*bs
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(motion.device)
        loss_ptm = F.cross_entropy(logits, ptm_labels)

        ##================= Motion to text ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            motion.device
        )
        # query_atts = torch.ones(query_tokens[:, 2:].size()[:-1], dtype=torch.long).to(
        #     motion.device
        # )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        loss_lm = lm_output.loss
        ##================= Text to Motion ========================##
        decoder_motion_ids = decoder_input_ids.clone()
        text_embeds = self.Qformer.bert.embeddings(decoder_motion_ids)
        text_embeds = text_embeds @ self.text_projection
        text_atts = text_tokens.attention_mask.clone()
        text_query_output = self.Qformer.bert(
            query_embeds=query_tokens,              # [bs, 49, 768]
            encoder_hidden_states=text_embeds,     # [bs, 49, 1408]
            encoder_attention_mask=text_atts,      # [bs, 49]
            use_cache=True,
            return_dict=True,
        )

        predicition = self.motion_cls(text_query_output.last_hidden_state)
        loss_fct = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
        # VVQ
        motion_targets = self.vq_model.encode(motion.permute(0, 2, 1))
        loss_gen = loss_fct(predicition.view(-1, self.num_tokens), motion_targets.view(-1))
        return QFormer_Output(
            loss = loss_lm + loss_ptm + loss_ptc + loss_gen,
            loss_ptc=loss_ptc,
            loss_ptm=loss_ptm,
            loss_lm=loss_lm,
            loss_gen=loss_gen
        ), text_feat, motion_re_feats
    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]
    @classmethod
    def from_config(cls, cfg):
        point_model = cfg.get("point_model", "ulip_point_bert")
        point_encoder_cfg = cfg.get("point_encoder_cfg")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)
        freeze_point_encoder = cfg.get("freeze_point_encoder", True)
        max_txt_len = cfg.get("max_txt_len", 32)
        ckpt_special_strs = cfg.get("ckpt_special_strs", None)

        model = cls(
            point_model=point_model,
            point_encoder_cfg=point_encoder_cfg,
            freeze_point_encoder=freeze_point_encoder,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            ckpt_special_strs=ckpt_special_strs
        )
        model.load_checkpoint_from_config(cfg)

        return model