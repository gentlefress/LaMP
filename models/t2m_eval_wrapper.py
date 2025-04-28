from models.t2m_eval_modules import *
from utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin
from models.q_former.QFormer_Base import QFormer_Base
import torch.nn.functional as F
def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        opt.output_emb_width = 512
        opt.down_t = 2
        opt.stride_t = 2
        opt.width = 512
        opt.depth = 3
        opt.dilation_growth_rate = 3
        opt.vq_norm = None
        print(opt)
        self.Qformer, self.query_tokens = QFormer_Base.init_Qformer(
            num_query_token=32, vision_width=1408, cross_attention_freq=2
        )
        self.query_tokens.requires_grad_(False)
        self.motion_encoder = QFormer_Base.init_motion_encoder("motion_encoder", opt, opt.dataset_name)
        self.tokenizer = QFormer_Base.init_tokenizer()
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.motion_projection = nn.Parameter(torch.empty(512, 1408)) 
        self.text_proj = nn.Linear(768, 512)
        self.motion_proj = nn.Linear(768, 512)
        self._prepare_qformer()
        # self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        # self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        # self.movement_encoder.to(opt.device)

        self.Qformer.to(opt.device)
        self.motion_projection.to(opt.device)
        self.query_tokens.to(opt.device)
        self.text_proj.to(opt.device)
        self.motion_proj.to(opt.device) 

        # self.text_encoder.eval()
        self.motion_encoder.eval()
        # self.movement_encoder.eval()
        self.text_proj.eval()
        self.motion_proj.eval()
        self.Qformer.eval()
    def _prepare_motion_encoder(self):
        ckpt = torch.load('/mnt/cap/karong/t2m/momask-codes/checkpoints/t2m/qformer_motion_b100_re_ep75_t2m/model/net_best_acc.tar', map_location='cpu')
        itm_ckpt = {}
        # motionproj_ckpt = {}
        for k, v in ckpt['motion_qformer'].items():
            if k.startswith('itm_head.'):
                itm_ckpt[k[9:]] = v
            # elif k.startswith('motion_proj.'):
            #     motionproj_ckpt[k.replace("motion_proj.", "")] = v
        self.itm_head.load_state_dict(itm_ckpt, strict=True)
        # self.motion_projection = nn.Parameter(ckpt['motion_qformer']['motion_projection'])
        # base_ckpt = {k.replace("motion_encoder.", ""): v for k,
        #                  v in ckpt['motion_qformer'].items()}
        # self.motion_encoder.load_state_dict(base_ckpt, strict=False)
    def _prepare_qformer(self):
        ckpt = torch.load('./pretrained/best.tar', map_location='cpu')
        # ckpt = torch.load('/mnt/cap/karong/t2m/momask-codes/checkpoints/t2m/qformer_motion_b100_re_ep75_t2m/model/net_best_acc.tar', map_location='cpu')
        # ckpt = torch.load('/mnt/cap/karong/t2m/momask-codes/checkpoints/t2m/qformer_motion_b70_re_ep200_t2m_rvq/model/net_best_acc.tar', map_location='cpu')
        # ckpt = torch.load('/mnt/cap/karong/t2m/momask-codes/checkpoints/t2m/qformer_motion_b100_re_ep200_t2m/model/net_best_acc.tar', map_location='cpu')
        base_ckpt = {k.replace("Qformer.", ""): v for k,
                    v in ckpt['motion_qformer'].items()}
        encoder_ckpt = {}
        textproj_ckpt = {}
        motionproj_ckpt = {}
        for k, v in ckpt['motion_qformer'].items():
            if k.startswith('motion_encoder.'):
                encoder_ckpt[k.replace("motion_encoder.", "")] = v
            elif k.startswith('text_proj.'):
                textproj_ckpt[k.replace("text_proj.", "")] = v
            elif k.startswith('motion_proj.'):
                motionproj_ckpt[k.replace("motion_proj.", "")] = v
        self.motion_encoder.load_state_dict(encoder_ckpt, strict=True)
        self.Qformer.load_state_dict(base_ckpt, strict=False)
        self.text_proj.load_state_dict(textproj_ckpt, strict=True)
        self.motion_proj.load_state_dict(motionproj_ckpt, strict=True)
        self.query_tokens = nn.Parameter(base_ckpt['query_tokens'])
        self.motion_projection = nn.Parameter(base_ckpt['motion_projection'])
    # Please note that the results does not follow the order of inputs
    def get_co_embeddings(self, caption, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()
            text_tokens = self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=32,
                return_tensors="pt",
            ).to(self.device)
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            text_embedding = self.text_proj(text_output.last_hidden_state[:, 0, :])
            # text_embedding = text_output.last_hidden_state[:, 0, :]
            movements = self.motion_encoder(motions.permute(0, 2, 1)).detach().permute(0, 2, 1)
            motion_atts = torch.ones(movements.size()[:-1], dtype=torch.long).to(             # [bs, 49]
                motions.device
            )
            motion_proj = self.motion_projection.to(self.device)
            motion_embeds = movements @ motion_proj
            query_tokens = self.query_tokens.expand(motion_embeds.shape[0], -1, -1).to(word_embs.device)              # [1, 32, 768] -> [bs, 32, 768]
            # query_tokens_text = self.query_tokens_text.expand(motion_embeds.shape[0], -1, -1)              # [1, 32, 768] -> [bs, 32, 768]
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,              # [bs, 49, 768]
                encoder_hidden_states=motion_embeds,     # [bs, 49, 1408]
                encoder_attention_mask=motion_atts,      # [bs, 49]
                use_cache=True,
                return_dict=True,
            )
            motion_embedding = torch.mean(self.motion_proj(query_output.last_hidden_state), dim=1)

            # text_atts_all = text_tokens.attention_mask
            # query_tokens_ptm = self.query_tokens.expand(motion_embeds.shape[0], -1, -1).to(motions.device)          # [3*bs, 32, 768]
            # query_atts_ptm = torch.ones(query_tokens_ptm.size()[:-1], dtype=torch.long).to(     # [3*bs, 32]
            #     motions.device
            # )
            # # query_tokens_ptm = self.query_tokens[:, 2:].expand(text_ids_all.shape[0], -1, -1)          # [3*bs, 32, 768]
            # # query_atts_ptm = torch.ones(query_tokens_ptm.size()[:-1], dtype=torch.long).to(     # [3*bs, 32]
            # #     motion.device
            # # )
            # attention_mask_all = torch.cat([query_atts_ptm, text_atts_all], dim=1)              # [3*bs, 32*2]

            # motion_atts_all = torch.ones(motion_embeds.size()[:-1], dtype=torch.long).to(     # [3*bs, 257]
            #     motions.device
            # )

            # output_ptm = self.Qformer.bert(
            #     text_tokens.input_ids,
            #     query_embeds=query_tokens_ptm,
            #     attention_mask=attention_mask_all,
            #     encoder_hidden_states=motion_embeds,
            #     encoder_attention_mask=motion_atts_all,
            #     return_dict=True,
            # )
  
            # motion_embedding = motion_re_feats

            '''Movement Encoding'''
            # movements = self.movement_encoder(motions[..., :-4]).detach()
            # m_lens = m_lens // self.opt.unit_length
            # motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            # text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            # text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not follow the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

## Borrowed form MDM
# our version
def build_evaluators(opt):
    movement_enc = MovementConvEncoder(opt['dim_pose']-4, opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
                                  pos_size=opt['dim_pos_ohot'],
                                  hidden_size=opt['dim_text_hidden'],
                                  output_size=opt['dim_coemb_hidden'],
                                  device=opt['device'])

    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])

    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc

# our wrapper
class EvaluatorWrapper(object):

    def __init__(self, dataset_name, device):
        opt = {
            'dataset_name': dataset_name,
            'device': device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_pos_ohot': len(POS_enumerator),
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': 263 if dataset_name == 'humanml' else 251,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': './checkpoints',
            'unit_length': 4,
        }

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_evaluators(opt)
        self.opt = opt
        self.device = opt['device']

        self.text_encoder.to(opt['device'])
        self.motion_encoder.to(opt['device'])
        self.movement_encoder.to(opt['device'])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
            # print(motions.shape, movements.shape, motion_embedding.shape, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = m_lens // self.opt['unit_length']
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding