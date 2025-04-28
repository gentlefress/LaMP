import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from models.lamp.QFormer_Base import QFormer_Base
from transformers import AutoTokenizer, OPTForCausalLM
from torch.cuda.amp import autocast as autocast
from models.vq.encdec import Encoder
class RoPEPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super(RoPEPositionEmbedding, self).__init__()
        self.dim = dim
        self.sinusoidal_embeddings = self.build_sinusoidal_embeddings(max_seq_len, dim)

    def build_sinusoidal_embeddings(self, max_seq_len, dim):
        sinusoidal_embeddings = torch.zeros(max_seq_len, dim)
        for pos in range(max_seq_len):
            for i in range(0, dim, 2):
                sinusoidal_embeddings[pos, i] = math.sin(pos / 10000 ** (2 * i / dim))
                sinusoidal_embeddings[pos, i + 1] = math.cos(pos / 10000 ** (2 * i / dim))
        return sinusoidal_embeddings

    def forward(self, x):
        seq_len, dim = x.size(1), x.size(2)
        pos_emb = self.sinusoidal_embeddings[:seq_len, :dim].unsqueeze(0).to(x.device)

        x_with_pos = torch.zeros_like(x)
        x_with_pos[:, :, 0::2] = x[:, :, 0::2] * pos_emb[:, :, 0::2].cos() - x[:, :, 1::2] * pos_emb[:, :, 1::2].sin()
        x_with_pos[:, :, 1::2] = x[:, :, 0::2] * pos_emb[:, :, 1::2].sin() + x[:, :, 1::2] * pos_emb[:, :, 0::2].cos()

        return x_with_pos
def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复

    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings
def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)


    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了



    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos

    return q, k
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)
def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)
# More on small value, less on large
def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low
class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                vq_model=None,
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq
        self.vq_model = vq_model
        self.Qformer, self.query_tokens = QFormer_Base.init_Qformer(
            num_query_token=49, vision_width=1408, cross_attention_freq=2
        )
        # 
        self.tokenizer = QFormer_Base.init_tokenizer()
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        self._prepare_qformer()
        for name, param in self.Qformer.named_parameters():
            param.requires_grad = False
    def get_block_size(self):
        return self.block_size
    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        return torch.cuda.amp.autocast(dtype=dtype)
    def _prepare_qformer(self):
        ckpt = torch.load('/mnt/cap/karong/t2m/momask-codes/checkpoints/t2m/qformer_motion_b100_re_ep75_t2m/model/net_best_acc.tar', map_location='cpu')
        # ckpt = torch.load('/mnt/cap/karong/t2m/momask-codes/checkpoints/kit/qformer_motion_b70_re_ep200_kit/model/best.tar', map_location='cpu')
        # ckpt = torch.load('../momask-codes/checkpoints/t2m/qformer_encnf_b100/model/net_best_acc .tar', map_location='cpu')
        base_ckpt = {k.replace("Qformer.", ""): v for k,
                    v in ckpt['motion_qformer'].items()}

        self.Qformer.load_state_dict(base_ckpt, strict=False)
        self.query_tokens = nn.Parameter(base_ckpt['query_tokens'])
    def _prepare_motion_encoder(self):
        ckpt = torch.load('/mnt/cap/karong/t2m/momask-codes/checkpoints/t2m/qformer_motion_b100_re_ep75_t2m/model/best.tar', map_location='cpu')
        itm_ckpt = {}
        # motionproj_ckpt = {}
        for k, v in ckpt['motion_qformer'].items():
            if k.startswith('itm_head.'):
                itm_ckpt[k[9:]] = v
            # elif k.startswith('motion_proj.'):
            #     motionproj_ckpt[k.replace("motion_proj.", "")] = v
        self.itm_head.load_state_dict(itm_ckpt, strict=True)
        self.motion_projection = nn.Parameter(ckpt['motion_qformer']['motion_projection'])
        base_ckpt = {k.replace("motion_encoder.", ""): v for k,
                         v in ckpt['motion_qformer'].items()}
        self.motion_encoder.load_state_dict(base_ckpt, strict=False)
    @torch.no_grad()
    def compute_confidence(self, text, motions):
        self._prepare_motion_encoder()
        movements = self.motion_encoder(motions.permute(0, 2, 1)).detach().permute(0, 2, 1)
        motion_atts = torch.ones(movements.size()[:-1], dtype=torch.long).to(             # [bs, 49]
            motions.device
        )
        motion_proj = self.motion_projection.to(motions.device)
        motion_embed = movements @ motion_proj
        query_tokens = self.query_tokens.expand(motion_embed.shape[0], -1, -1).to(motions.device)              # [1, 32, 768] -> [bs, 32, 768]
        # # query_tokens_text = self.query_tokens_text.expand(motion_embeds.shape[0], -1, -1)              # [1, 32, 768] -> [bs, 32, 768]
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        ).to(motion_embed.device)
        query_tokens_ptm = self.query_tokens.expand(motion_embed.shape[0], -1, -1)          # [3*bs, 32, 768]
        query_atts_ptm = torch.ones(query_tokens_ptm.size()[:-1], dtype=torch.long).to(     # [3*bs, 32]
            motion_embed.device
        )
        attention_mask_all = torch.cat([query_atts_ptm, text_tokens.attention_mask], dim=1)              # [3*bs, 32*2]

        motion_atts_all = torch.ones(motion_embed.size()[:-1], dtype=torch.long).to(     # [3*bs, 257]
            motion_embed.device
        )
        output_ptm = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens_ptm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=motion_embed,
            encoder_attention_mask=motion_atts_all,
            return_dict=True,
        )
        vl_embeddings = output_ptm.last_hidden_state[:, : query_tokens_ptm.size(1), :]      # [3*bs, 32, 768]
        vl_output = self.itm_head(vl_embeddings)                                            # [3*bs, 32, 768] -> [3*bs, 32, 2]
        logits = vl_output.mean(dim=1)      
        return logits
    def forward(self, idxs, clip_feature, training=True):
        # query_tokens = self.query_tokens.expand(clip_feature.shape[0], -1, -1)
        text_tokens = self.tokenizer(
            clip_feature,
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt",
        ).to('cuda:0')
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        # text features to Q-Former, projection is not initilized
        # text_feature = text_output.last_hidden_state
        text_feature = text_output.last_hidden_state[:, 0, :]
        bs = text_feature.size(0)
        # if training:
        #     mask = torch.bernoulli(torch.ones(bs, device=clip_feature.device) * 0.1).view(bs, 1)
        #     clip_feature = clip_feature * (1. - mask)
        feat = self.trans_base(idxs, text_feature)
        logits = self.trans_head(feat)
        return logits
        # cycle loss generated motion to text
        # logits, x = self.trans_head(feat)
        # if not training:
        #     return logits
        # pred_logits = self.proj_2_motion(x[:, 1:])
        # probs = torch.softmax(pred_logits, dim=-1)
        # dist = Categorical(probs)
        # cls_pred_index = dist.sample()
        # pred_motion = self.vq_model.forward_decoder(cls_pred_index[:, :-1])
        # pred_motion = pred_motion.view(bs, -1, pred_motion.size(-1))
        # opt_input = self.vq_model.vqvae.encoder(pred_motion.permute(0, 2, 1)).permute(0, 2, 1)
        # opt_input = self.motion_qformer(opt_input)
        # motion_atts = torch.ones(opt_input.size()[:-1], dtype=torch.long).to(
        #     x.device
        # )
        # query_tokens = self.query_tokens.expand(opt_input.shape[0], -1, -1)
        # query_output = self.Qformer.bert(
        #     query_embeds=query_tokens,
        #     encoder_hidden_states=opt_input,
        #     encoder_attention_mask=motion_atts,
        #     return_dict=True,
        # )
        # inputs_opt = self.opt_proj(query_output.last_hidden_state)
        # atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(x.device)
        # self.opt_tokenizer.padding_side = "right"
        # text = [t + "\n" for t in clip_feature]
        # self.prompt = 'a man is'
        # prompt = self.prompt
        # prompt = [prompt] * logits.size(0)
        # opt_tokens = self.opt_tokenizer(
        #     prompt,
        #     # text,
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=32,
        # ).to(x.device)
        # targets = opt_tokens.input_ids.masked_fill(
        #     opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        # )
        # empty_targets = (
        #     torch.ones(atts_opt.size(), dtype=torch.long).to(x.device).fill_(-100)
        # )
        # targets = torch.cat([empty_targets, targets], dim=1)
        # inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        # inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        # attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        # with self.maybe_autocast():
        #     outputs = self.opt_model.generate(
        #             inputs_embeds=inputs_embeds, 
        #             attention_mask=attention_mask,
        #             do_sample=False,
        #             top_p=0.9,
        #             temperature=1,
        #             num_beams=5,
        #             max_length=32,
        #             min_length=1,
        #             eos_token_id=self.eos_token_id,
        #             repetition_penalty=1.0,
        #             length_penalty=1.0,
        #             num_return_sequences=1,
        #         )
        # output_text = self.opt_tokenizer.batch_decode(
        #         outputs, skip_special_tokens=True
        #     )
        # output_text = [text.strip() for text in output_text]
        # with self.maybe_autocast():
        #     outputs = self.opt_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         return_dict=True,
        #         labels=targets,
        #     )
        # loss = outputs.loss
        # return logits, loss

    def sample(self, clip_feature, if_categorial=False):
        # Q-former utilize bert encoder to get conditional embedding
        for k in range(self.block_size):
            if k == 0:
                x = []
            else:
                x = xs
            logits = self.forward(x, clip_feature, training=False)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample()
                if idx == self.num_vq:
                    break
                idx = idx.unsqueeze(-1)
            else:
                _, idx = torch.topk(probs, k=1, dim=-1)
                if idx[0] == self.num_vq:
                    break
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)
            
            if k == self.block_size - 1:
                return xs[:, :-1]
        return xs

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # attn_cond = torch.ones(83, 32).to(k.device)
        # attn_moti = torch.zeros(32, 51).to(k.device)
        # attn_mask = torch.cat([attn_cond.to(k.device), torch.cat([attn_moti.to(k.device), self.mask], dim=0)], dim=1).view(1, 1, 83, 83)
        
        # Apply RoPE
        # q, k = RoPE(q, k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(attn_mask[:,:,:T,:T] == 0, float('-inf'))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
class CrossCondTransBase(nn.Module): 

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        # CVQ
        # self.tok_emb = nn.Embedding(num_vq * 2 + 2, embed_dim)
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        # self.cond_emb = nn.Linear(768, embed_dim)
        # self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        # self.pos_embed = pos_encoding.PositionEmbedding(83, embed_dim, 0.0, False)
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)
        # self.pos_embed = RoPEPositionEmbedding(embed_dim, block_size)
        # self.sos = torch.nn.Parameter(torch.zeros(1, embed_dim))
        # nn.init.normal_(self.sos)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature):
        # sos = self.sos.expand(clip_feature.size(0), -1, -1)
        if len(idx) == 0:
            # token_embeddings = self.cond_emb(clip_feature)
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
            # token_embeddings = torch.cat([token_embeddings, sos], dim=1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            # token_embeddings = torch.cat([self.cond_emb(clip_feature), sos, token_embeddings], dim=1)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            
        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        # CVQ
        # self.head = nn.Linear(embed_dim, num_vq * 2 + 1, bias=False)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
        # return logits, x

    


        

