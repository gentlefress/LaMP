import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.m2t.pos_encoding as pos_encoding
from models.lamp.QFormer_Base import QFormer_Base
from transformers import AutoTokenizer, OPTForCausalLM
from torch.cuda.amp import autocast as autocast
from models.m2t.encdec import Encoder
from models.m2t.llama import *
from models.m2t.lora import *
from models.m2t.tokenizer import *
from peft import LoraConfig, get_peft_model
def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
    return loss
class Motion2Text_Transformer(nn.Module):

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
        # self.Qformer, self.query_tokens = QFormer_Base.init_Qformer(
        #     num_query_token=32, vision_width=1408, cross_attention_freq=2
        # )
        self.Qformer, self.query_tokens = QFormer_Base.init_Qformer(
            num_query_token=49, vision_width=1408, cross_attention_freq=2
        )
        # confidence metric
        self.motion_encoder = Encoder(263, 512, 2, 2, 512, 3, 3, activation='relu', norm=None)
        self.motion_projection = nn.Parameter(torch.empty(512, 1408))
        self.tokenizer = QFormer_Base.init_tokenizer()
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self._prepare_qformer()
        for name, param in self.motion_encoder.named_parameters():
            param.requires_grad = False
        self.opt_tokenizer = AutoTokenizer.from_pretrained("./opt2.7b", use_fast=False)
        self.opt_model = OPTForCausalLM.from_pretrained(
            "./opt2.7b", torch_dtype=torch.float16
        )
        self.eos_token_id = self.opt_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, 2560
        )
        config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        )
        self.llm = get_peft_model(self.opt_model, config)
    def get_block_size(self):
        return self.block_size
    def maybe_autocast(self, dtype=torch.float32):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        return torch.cuda.amp.autocast(dtype=dtype)
    def _prepare_qformer(self):
        ckpt = torch.load('./pretrained/best.tar', map_location='cpu')

        base_ckpt = {k.replace("Qformer.", ""): v for k,
                    v in ckpt['motion_qformer'].items()}
        self.motion_projection = nn.Parameter(ckpt['motion_qformer']['motion_projection'])
        self.Qformer.load_state_dict(base_ckpt, strict=False)
        self.query_tokens = nn.Parameter(base_ckpt['query_tokens'])
        base_ckpt = {k.replace("motion_encoder.", ""): v for k,
                    v in ckpt['motion_qformer'].items()}
        self.motion_encoder.load_state_dict(base_ckpt, strict=False)
    def forward(self, motion, clip_feature, training=True):
        bs = len(clip_feature)
        # with self.maybe_autocast():
        motion_embeds = self.motion_encoder(motion.permute(0, 2, 1)).permute(0, 2, 1)
        motion_embeds = motion_embeds @ self.motion_projection
        
        motion_attns = torch.ones(motion_embeds.size()[:-1], dtype=torch.long).to(
            motion.device
        )
        query_tokens = self.query_tokens.expand(motion_embeds.shape[0], -1, -1)
        with self.maybe_autocast():
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=motion_embeds,
                encoder_attention_mask=motion_attns,
                return_dict=True,
            )

        self.opt_tokenizer.padding_side = "right"
        text = [t + "\n" for t in clip_feature]

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(motion.device)
        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=32,
        ).to(motion.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(motion.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        inputs_embeds = self.llm.model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)

        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss
        return {"loss": loss}
    @torch.no_grad()
    def generate(
        self,
        motion,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=100,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        with self.maybe_autocast():
            motion_embeds = self.motion_encoder(motion.permute(0, 2, 1)).permute(0, 2, 1)                                            # [bs, 8192, 3] -> [bs, 512, 384]
            motion_embeds = motion_embeds @ self.motion_projection                                    # [bs, 512, 384] -> [bs, 512, 768]  
            motion_atts = torch.ones(motion_embeds.size()[:-1], dtype=torch.long).to(
                motion.device
            )
            query_tokens = self.query_tokens.expand(motion_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=motion_embeds,
                encoder_attention_mask=motion_atts,
                return_dict=True,
            )

            # inputs_opt = self.llama_proj(query_output.last_hidden_state)
            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                motion.device
            )
            self.prompt = 'a man is'
            prompt = self.prompt
            prompt = [prompt] * motion.size(0)

            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length,
            ).to(motion.device)
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            # new version for transformers>=4.27
            inputs_embeds = self.llm.get_input_embeddings()(opt_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_opt, inputs_embeds],dim=1)
            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.opt_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text

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
        # self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.cond_emb = nn.Linear(768, embed_dim)
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

    


        

