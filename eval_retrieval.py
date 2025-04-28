import os
from os.path import join as pjoin

import torch

from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
import models.mask_transformer.t2m_trans as trans
from models.vq.model import RVQVAE
import models.vq.vqvae as vqvae
from options.eval_option import EvalT2MOptions
from utils.get_opt import get_opt
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper

import utils.eval_t2m as eval_t2m
from utils.fixseed import fixseed

import numpy as np

def load_vq_model(vq_opt):
    # opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_model = vqvae.HumanVQVAE(vq_opt, ## use args to define different parameters in different quantizers
                        vq_opt.nb_code,
                        vq_opt.code_dim,
                        vq_opt.output_emb_width,
                        vq_opt.down_t,
                        vq_opt.stride_t,
                        vq_opt.width,
                        vq_opt.depth,
                        vq_opt.dilation_growth_rate,
                        vq_opt.vq_act,
                        vq_opt.vq_norm)
    # ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
    #                         map_location=opt.device)
    # ckpt = torch.load('/mnt/cap/karong/t2m/pretrained/VQVAE_KIT/net_best_fid.pth', map_location='cpu')
    ckpt = torch.load('./pretrained/net_last.pth', map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt

if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    dim_pose = 251 if opt.dataset_name == 'kit' else 263

    # out_dir = pjoin(opt.check)
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    model_dir = pjoin(root_dir, 'model')
    out_dir = pjoin(root_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    out_path = pjoin(out_dir, "%s.log"%opt.ext)

    f = open(pjoin(out_path), 'w')

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)
    clip_version = 'ViT-B/32'

    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # res_opt_path = ''
    # res_opt = get_opt(res_opt_path, device=opt.device)
    # res_model = load_res_model()

    # assert res_opt.vq_name == model_opt.vq_name

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if opt.dataset_name == 'kit' \
        else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 21 if opt.dataset_name == 'kit' else 22

    eval_val_loader, eval_val_dataset = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    top5 = []
    top10 = []
    matching = []
    mm = []

    repeat_time = 20
    for i in range(repeat_time):
        with torch.no_grad():
            best_div, Rprecision, best_matching = \
                eval_t2m.evaluation_qformer_retrieval(out_dir, eval_val_loader, eval_val_dataset, i, eval_wrapper=eval_wrapper)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        top5.append(Rprecision[4])
        top10.append(Rprecision[9])
        matching.append(best_matching)

    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    top5 = np.array(top5)
    top10 = np.array(top10)
    matching = np.array(matching)


    msg_final = f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}, TOP5. {np.mean(top5):.3f}, conf. {np.std(top5) * 1.96 / np.sqrt(repeat_time):.3f}, TOP10. {np.mean(top10):.3f}, conf. {np.std(top10) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n"
    # logger.info(msg_final)
    print(msg_final)


# python eval_t2m_trans.py --name t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_vq --dataset_name t2m --gpu_id 3 --cond_scale 4 --time_steps 18 --temperature 1 --topkr 0.9 --gumbel_sample --ext cs4_ts18_tau1_topkr0.9_gs