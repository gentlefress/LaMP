import os 
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
# import clip
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_train
from dataset import dataset_TM_eval
from dataset import dataset_tokenize
import models.m2t_trans_llama as trans
# import models.m2t_trans as trans
from options.get_eval_option import get_opt, get_rvq_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings('ignore')
import bert_score
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
from contextlib import contextmanager
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad
def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

##### ---- Dataloader ---- #####
train_loader_token = dataset_tokenize.DATALoader(args.dataname, 32, unit_length=2**args.down_t)

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
# val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 30, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####


net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)


trans_encoder = trans.Motion2Text_Transformer(num_vq=args.nb_code, 
                                vq_model=net,
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
net.load_state_dict(ckpt[model_key])
# net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

train_loader_iter = dataset_tokenize.cycle(train_loader_token)

##### ---- Training ---- #####
eval_trans.evaluation_m2t(args.out_dir, net, trans_encoder, logger, writer, 0, best_precision=0, best_iter=0, best_recall=0, best_f1=0, eval_wrapper=eval_wrapper)
while nb_iter <= args.total_iter:
    
    batch = next(train_loader_iter)
    motion, name, clip_text = batch
    motion = motion.cuda().float()
    bs = motion.size(0)
    # m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
    # target = m_tokens    # (bs, 26)
    # target = target.cuda()
    
    # text = clip.tokenize(clip_text, truncate=True).cuda()
    
    # feat_clip_text = clip_model.encode_text(text).float()

    # input_index = target[:,:-1]
    
    loss = trans_encoder(motion, clip_text, training=True)
    loss_cls = loss["loss"]
    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()
    # P, R, F1 = bert_score.score(cands, refs, lang="en", verbose=True)
    # print(f'Precision: {P.mean():.3f}, Recall: {R.mean():.3f}, F1: {F1.mean():.3f}')
    avg_loss_cls = avg_loss_cls + loss_cls.item()
    # nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()



    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}"
        logger.info(msg)
        avg_loss_cls = 0.
        # nb_sample_train = 0

    if nb_iter % args.eval_iter ==  0:
        # best_iter, best_precision, best_recall, best_f1, writer, logger = eval_trans.evaluation_m2t(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_precision, best_iter, best_recall, best_f1, eval_wrapper=eval_wrapper)
        # eval_trans.evaluation_m2t(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_precision=0, best_iter=0, best_recall=0, best_f1=0, eval_wrapper=eval_wrapper)
        eval_trans.evaluation_m2t(args.out_dir, net, trans_encoder, logger, writer, 0, best_precision=0, best_iter=0, best_recall=0, best_f1=0, eval_wrapper=eval_wrapper)
    if nb_iter == args.total_iter: 
        # msg_final = f"Train. Iter {best_iter} : Precision. {best_precision:.5f}, Recall. {best_recall:.4f}, F1. {best_f1:.4f}"
        # logger.info(msg_final)
        msg_final = f"Finish training"
        logger.info(msg_final)
        break            