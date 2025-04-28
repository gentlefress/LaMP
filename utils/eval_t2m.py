import os

# import clip
import numpy as np
import torch
# from scipy import linalg
from utils.metrics import *
# from utils.bodypart_metrics import *
import torch.nn.functional as F
# import visualization.plot_3d_global as plot_3d
from utils.plot_script import plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
from utils.motion_process import recover_from_ric
from os.path import join as pjoin
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#
# def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
#     xyz = xyz[:1]
#     bs, seq = xyz.shape[:2]
#     xyz = xyz.reshape(bs, seq, -1, 3)
#     plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(), title_batch, outname)
#     plot_xyz = np.transpose(plot_xyz, (0, 1, 4, 2, 3))
#     writer.add_video(tag, plot_xyz, nb_iter, fps=20)
def plot_t2m(eval_dataset, data, save_dir):
    data = eval_dataset.inv_transform(data)
    for i in range(len(data)):
        joint_data = data[i]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
        save_path = pjoin(save_dir, '%02d.mp4' % (i))
        plot_3d_motion(save_path, t2m_kinematic_chain, joint, title="None", fps=20, radius=4)
@torch.no_grad()        
def evaluation_vqvae_bodypart(out_dir, val_loader, net, writer, nb_iter, best_fid, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 
    """
    Evaluate the VQVAE, used in train and test.
    Compute the FID, DIV, and R-Precision.
    """
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # Get motion and parts. We use parts to represent parts' motion.
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, parts = batch
        motion = motion.cuda()
        if torch.isnan(motion).sum() > 0 or torch.isinf(motion).sum() > 0:
            print('Detected NaN or Inf in raw motion data')
            print('NaN elem numbers:', torch.isnan(motion).sum())
            print('Inf elem numbers:', torch.isinf(motion).sum())
            print('motion:', motion)
        # (text, motion) ==> (text_emb, motion_emb)
        #   motion is normalized.
        et, em = eval_wrapper.get_co_embeddings(caption, word_embeddings, pos_one_hots, sent_len, motion, m_length)

        if torch.isnan(em).sum() > 0:
            print('Detected NaN in em (embedding of motion), replace NaN with 0.0')
            print('NaN elem numbers:', torch.isnan(em).sum())
            print('em:', em)
            em = torch.nan_to_num(em)  # use default param to replace nan with 0.0. Require pytorch >= 1.8.0


        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):

            # [gt_motion] (augmented representation) ==[de-norm, convert]==> [gt_xyz] (xyz representation)
            # pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            # pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


            # Preprocess parts: get single sample from the batch
            single_parts = []
            for p in parts:
                single_parts.append(p[i:i+1, :m_length[i]].cuda())

            # (parts, GT) ==> (reconstruct_parts)
            #   parts is normalized.
            pred_pose, loss, perplexity = net(single_parts)
            # pred_parts ==> whole_motion
            #   todo: support different shared_joint_rec_mode in the parts2whole function

            # de-normalize reconstructed motion
            # pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())

            # Convert to xyz representation
            #   todo: maybe we should support the recover_from_rot, not only ric.
            # pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            # if savenpy:
            #     np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
            #     np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            # if i < min(4, bs):
            #     draw_org.append(pose_xyz)
            #     draw_pred.append(pred_xyz)
            #     draw_text.append(caption[i])

        # pred_pose_eval is normalized
        et_pred, em_pred = eval_wrapper.get_co_embeddings(caption,
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        # if nb_iter % 5000 == 0 : 
        #     for ii in range(4):
        #         tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid = fid
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_vqvae(out_dir, val_loader, net, writer, ep, best_fid, best_div, best_top1,
                     best_top2, best_top3, best_matching, eval_wrapper, save=True, draw=True):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        # print(len(batch))
        # word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, motion2d = batch
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        # motion = motion.cuda()
        motion = motion.cuda()[:, :-4]
        # motion2d = motion2d.cuda()
        et, em = eval_wrapper.get_co_embeddings(caption, word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        # pred_pose_eval, loss_commit, code_idx = net(motion, mode='eval', temperature=1.)

        # pred_pose_eval = pred_pose_eval.permute(0, 2, 1)
        # pred_pose_eval, loss_commit, perplexity = net(motion, text=None)
        pred_pose_eval, loss_commit, perplexity = net(motion)
        # pred_pose_eval, _, loss_commit = net(motion2d)
        loss_commit = sum(loss_commit)
        perplexity = sum(perplexity)
        pred_pose_eval = pred_pose_eval[0]
        et_pred, em_pred = eval_wrapper.get_co_embeddings(caption, word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Ep %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_score_real. %.4f, matching_score_pred. %.4f"%\
          (ep, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred )
    # logger.info(msg)
    print(msg)

    if draw:
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)

    if fid < best_fid:
        msg = "--> --> \t FID Improved from %.5f to %.5f !!!" % (best_fid, fid)
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_fid.tar'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = "--> --> \t Diversity Improved from %.5f to %.5f !!!"%(best_div, diversity)
        if draw: print(msg)
        best_div = diversity
        # if save:
        #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1:
        msg = "--> --> \t Top1 Improved from %.5f to %.5f !!!" % (best_top1, R_precision[0])
        if draw: print(msg)
        best_top1 = R_precision[0]
        # if save:
        #     torch.save({'vq_model': net.state_dict(), 'ep':ep}, os.path.join(out_dir, 'net_best_top1.tar'))

    if R_precision[1] > best_top2:
        msg = "--> --> \t Top2 Improved from %.5f to %.5f!!!" % (best_top2, R_precision[1])
        if draw: print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = "--> --> \t Top3 Improved from %.5f to %.5f !!!" % (best_top3, R_precision[2])
        if draw: print(msg)
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from %.5f to %.5f !!!" % (best_matching, matching_score_pred)
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, os.path.join(out_dir, 'net_best_mm.tar'))

    # if save:
    #     torch.save({'net': net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_vqvae_plus_mpjpe(eval_dataset, val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(None, word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity, all_out = net(motion)
        for i in range(len(all_out)):
            data = torch.cat([motion[:4], all_out[i][:4]], dim=0).detach().cpu().numpy()
            save_dir = pjoin('/mnt/cap/karong/t2m/output/visual', 'E%04d' % (i))
            os.makedirs(save_dir, exist_ok=True)
            plot_t2m(eval_dataset, data, save_dir)

        # pred_pose_eval, loss_commit, perplexity, all = net(motion)


        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)

            mpjpe += torch.sum(calculate_mpjpe(gt, pred))
            # print(calculate_mpjpe(gt, pred).shape, gt.shape, pred.shape)
            num_poses += gt.shape[0]

        # print(mpjpe, num_poses)
        # exit()

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    mpjpe = mpjpe / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, MPJPE. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, mpjpe)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, mpjpe

@torch.no_grad()
def evaluation_vqvae_plus_l1(val_loader, net, repeat_id, eval_wrapper, num_joint):
    net.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        pred_pose_eval, loss_commit, perplexity = net(motion)
        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_res_plus_l1(val_loader, vq_model, res_model, repeat_id, eval_wrapper, num_joint, do_vq_res=True):
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    l1_dist = 0
    num_poses = 1
    for batch in val_loader:
        # print(len(batch))
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        caption = caption.cuda()
        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(caption, word_embeddings, pos_one_hots, sent_len, motion, m_length)
        # et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        # num_joints = 21 if motion.shape[-1] == 251 else 22

        # pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        if do_vq_res:
            code_ids, all_codes = vq_model.encode(motion)
            if len(code_ids.shape) == 3:
                pred_vq_codes = res_model(code_ids[..., 0])
            else:
                pred_vq_codes = res_model(code_ids)
            # pred_vq_codes = pred_vq_codes - pred_vq_res + all_codes[1:].sum(0)
            pred_pose_eval = vq_model.decoder(pred_vq_codes)
        else:
            rec_motions, _, _ = vq_model(motion)
            pred_pose_eval = res_model(rec_motions)        # all_indices,_  = net.encode(motion)
        # pred_pose_eval = net.forward_decoder(all_indices[..., :1])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval,
                                                          m_length)

        bgt = val_loader.dataset.inv_transform(motion.detach().cpu().numpy())
        bpred = val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())
        for i in range(bs):
            gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
            pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
            # gt = motion[i, :m_length[i]]
            # pred = pred_pose_eval[i, :m_length[i]]
            num_pose = gt.shape[0]
            l1_dist += F.l1_loss(gt, pred) * num_pose
            num_poses += num_pose

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f"%\
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0],R_precision_real[1], R_precision_real[2],
           R_precision[0],R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist

@torch.no_grad()
def evaluation_mask_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False):

    def save(file_name, ep):
        t2m_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            't2m_transformer': t2m_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    correct = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        # word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, motion2d = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()
        # confident = torch.zeros((bs, bs)).cuda()
        # for k in range(bs):
        # mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
        # pred_motion = trans.generate(vq_model, clip_text, bs, None)
        # mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
        # pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        # pred_len = torch.ones(bs).long()
        # for k in range(bs):
        #     pred_pose = vq_model.forward_decoder(mids[k][:(m_length // 4)[k]])
        #     cur_len = pred_pose.shape[1]
        #     pred_len[k] = min(cur_len, seq)
        #     pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
        for k in range(bs):
            index_motion = trans.autoregressive_infer_cfg(clip_text[k:k+1], 1, m_length[k:k+1], g_seed=0, cfg=4, top_k=450, top_p=0.95) 
            # index_motion = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
            index_motion = torch.cat([index_motion, index_motion[:, -1].unsqueeze(1)], dim=-1)
            x_d = vq_model.vqvae.quantizer.dequantize(index_motion)
            x_d = x_d.view(1, -1, vq_model.vqvae.code_dim).permute(0, 2, 1).contiguous()
            x = vq_model.vqvae.decoder.model[0](x_d)
            x = vq_model.vqvae.decoder.model[1](x)
            x = vq_model.vqvae.decoder.model[6](x)
            x = vq_model.vqvae.decoder.model[7](x)
            x = vq_model.vqvae.decoder.model[-3](x)
            x = vq_model.vqvae.decoder.model[-2](x)
            pred_pose = vq_model.vqvae.decoder.model[-1](x).permute(0, 2, 1)
            # pred_pose = vq_model.forward_decoder(index_motion[:m_length[k:k+1] // 4])
            cur_len = pred_pose.shape[1]
            pred_len[k] = min(cur_len, seq)
            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
        # mids.unsqueeze_(-1)
        # pred_motions = vq_model.forward_decoder(mids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len, pred_pose_eval.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_pose_eval[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)

    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer

@torch.no_grad()
def evaluation_res_transformer(out_dir, val_loader, trans, vq_model, writer, ep, best_fid, best_div,
                           best_top1, best_top2, best_top3, best_matching, eval_wrapper, plot_func,
                           save_ckpt=False, save_anim=False, cond_scale=2, temperature=1):

    def save(file_name, ep):
        res_trans_state_dict = trans.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            # 'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
        }
        torch.save(state, file_name)

    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # (b, seqlen)
        if ep == 0:
            pred_ids = code_indices[..., 0:1]
        else:
            pred_ids = trans.generate(code_indices[..., 0], clip_text, m_length//4,
                                      temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Ep {ep} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    # if draw:
    writer.add_scalar('./Test/FID', fid, ep)
    writer.add_scalar('./Test/Diversity', diversity, ep)
    writer.add_scalar('./Test/top1', R_precision[0], ep)
    writer.add_scalar('./Test/top2', R_precision[1], ep)
    writer.add_scalar('./Test/top3', R_precision[2], ep)
    writer.add_scalar('./Test/matching_score', matching_score_pred, ep)


    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid, best_ep = fid, ep
        if save_ckpt:
            save(os.path.join(out_dir, 'model', 'net_best_fid.tar'), ep)

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        print(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        print(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        print(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        print(msg)
        best_top3 = R_precision[2]

    if save_anim:
        rand_idx = torch.randint(bs, (3,))
        data = pred_motions[rand_idx].detach().cpu().numpy()
        captions = [clip_text[k] for k in rand_idx]
        lengths = m_length[rand_idx].cpu().numpy()
        save_dir = os.path.join(out_dir, 'animation', 'E%04d' % ep)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        plot_func(data, save_dir, captions, lengths)


    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer


@torch.no_grad()
def evaluation_res_transformer_plus_l1(val_loader, vq_model, trans, repeat_id, eval_wrapper, num_joint,
                                       cond_scale=2, temperature=1, topkr=0.9, cal_l1=True):


    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda().long()
        pose = pose.cuda().float()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        code_indices, all_codes = vq_model.encode(pose)
        # print(code_indices[0:2, :, 1])

        pred_ids = trans.generate(code_indices[..., 0], clip_text, m_length//4, topk_filter_thres=topkr,
                                  temperature=temperature, cond_scale=cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)

        pred_motions = vq_model.forward_decoder(pred_ids)

        if cal_l1:
            bgt = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
            bpred = val_loader.dataset.inv_transform(pred_motions.detach().cpu().numpy())
            for i in range(bs):
                gt = recover_from_ric(torch.from_numpy(bgt[i, :m_length[i]]).float(), num_joint)
                pred = recover_from_ric(torch.from_numpy(bpred[i, :m_length[i]]).float(), num_joint)
                # gt = motion[i, :m_length[i]]
                # pred = pred_pose_eval[i, :m_length[i]]
                num_pose = gt.shape[0]
                l1_dist += F.l1_loss(gt, pred) * num_pose
                num_poses += num_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motions.clone(),
                                                          m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist


@torch.no_grad()
def evaluation_mask_transformer_test(val_loader, vq_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0
    for i, batch in enumerate(val_loader):
        # print(i)
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        m_length = m_length.cuda()
        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        # if i < num_mm_batch:
        # # (b, seqlen, c)
        #     motion_multimodality_batch = []
        #     for _ in range(30):
        #         pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        #         pred_len = torch.ones(bs).long()
        #         pred_ids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
        #                 temperature=temperature, topk_filter_thres=topkr,
        #                 gsample=gsample, force_mask=force_mask)
        #             # mids = trans.generate(clip_text, m_length//4, time_steps, cond_scale, temperature=1)
        #         for k in range(bs):
        #             # index_motion = trans.sample(clip_text[k:k+1], False)
        #             # try:
        #             # pred_ids = trans.generate(clip_text[k:k+1], (m_length // 4)[k:k+1], time_steps, cond_scale,
        #             #     temperature=temperature, topk_filter_thres=topkr,
        #             #     gsample=gsample, force_mask=force_mask)
        #             # index_motion = trans.sample(feat_clip_text[k:k+1], False)
        #             # except:
        #             #     pred_ids = torch.ones(1,1).cuda().long()
        #             # RVQ
        #             # index_motion.unsqueeze_(-1)
        #             # pred_pose = vq_model.forward_decoder(pred_ids)
        #             pred_pose = vq_model.forward_decoder(pred_ids[k][:(m_length // 4)[k]])
        #             cur_len = pred_pose.shape[1]
        #             pred_len[k] = min(cur_len, seq)
        #             pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
        #         # mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
        #         #                       temperature=temperature, topk_filter_thres=topkr,
        #         #                       gsample=gsample, force_mask=force_mask)

        #         # # motion_codes = motion_codes.permute(0, 2, 1)
        #         # mids.unsqueeze_(-1)
        #         # pred_motions = vq_model.forward_decoder(mids)

        #         et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len, pred_pose_eval.clone(),
        #                                                           m_length)
        #         # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
        #         motion_multimodality_batch.append(em_pred.unsqueeze(1))
        #     motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
        #     motion_multimodality.append(motion_multimodality_batch)
        # else:
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()
        pred_ids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
                    temperature=temperature, topk_filter_thres=topkr,
                    gsample=gsample, force_mask=force_mask)
        for k in range(bs):
                # index_motion = trans.sample(clip_text[k:k+1], False)
                # try:
                # pred_ids = trans.generate(clip_text[k:k+1], (m_length // 4)[k:k+1], time_steps, cond_scale,
                #     temperature=temperature, topk_filter_thres=topkr,
                #     gsample=gsample, force_mask=force_mask)
                # index_motion = trans.sample(feat_clip_text[k:k+1], False)
                # except:
                #     pred_ids = torch.ones(1,1).cuda().long()
                # RVQ
                # index_motion.unsqueeze_(-1)
                # pred_ids = trans.generate(clip_text[k:k+1], (m_length // 4)[k], time_steps, cond_scale,
                #     temperature=temperature, topk_filter_thres=topkr,
                #     gsample=gsample, force_mask=force_mask)
                # pred_pose = vq_model.forward_decoder(pred_ids)
            pred_pose = vq_model.forward_decoder(pred_ids[k][:(m_length // 4)[k]])
            cur_len = pred_pose.shape[1]
            pred_len[k] = min(cur_len, seq)
            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            # mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
            #                       temperature=temperature, topk_filter_thres=topkr,
            #                       force_mask=force_mask)

            # # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            # pred_motions = vq_model.forward_decoder(mids)

        et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len,
                                                            pred_pose_eval.clone(),
                                                            m_length)
        data = pred_pose_eval.detach().cpu().numpy()
        captions = clip_text
        save_dir = os.path.join('./demo', 'animation', 'E%04d' % 3)
        os.makedirs(save_dir, exist_ok=True)
        # print(lengths)
        data = val_loader.dataset.inv_transform(data)
        # print(ep_curves.shape)
        for i, (caption, joint_data) in enumerate(zip(captions, data)):
            joint_data = joint_data[:196]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
            save_path = pjoin(save_dir, name[i]+'.mp4')
            save_npy = pjoin(save_dir, name[i]+'.npy')
            np.save(save_npy, joint_data)            
            plot_3d_motion(save_path, t2m_kinematic_chain, joint, title=caption, fps=20)
        pose = pose.cuda().float()
        et, em = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality


@torch.no_grad()
def evaluation_mask_transformer_test_plus_res(val_loader, vq_model, res_model, trans, repeat_id, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False,
                                              cal_mm=True, res_cond_scale=5):
    trans.eval()
    vq_model.eval()
    res_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if force_mask or (not cal_mm):
        num_mm_batch = 0
    else:
        num_mm_batch = 3

    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        # for i in range(mm_batch)
        if i < num_mm_batch:
        # (b, seqlen, c)
            motion_multimodality_batch = []
            for _ in range(30):
                pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
                pred_len = torch.ones(bs).long()
                for k in range(bs):
                    # index_motion = trans.sample(clip_text[k:k+1], False)
                    # try:
                    index_motion = res_model.sample(clip_text[k:k+1], False)
                    pred_ids = trans.generate(clip_text[k:k+1], index_motion, index_motion.size(-1), time_steps, cond_scale,
                        temperature=temperature, topk_filter_thres=topkr,
                        gsample=gsample, force_mask=force_mask)
                    # index_motion = trans.sample(feat_clip_text[k:k+1], False)
                    # except:
                    #     pred_ids = torch.ones(1,1).cuda().long()
                    # RVQ
                    # index_motion.unsqueeze_(-1)
                    pred_pose = vq_model.forward_decoder(pred_ids)
                    cur_len = pred_pose.shape[1]
                    pred_len[k] = min(cur_len, seq)
                    pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                # motion_codes = motion_codes.permute(0, 2, 1)
                # mids.unsqueeze_(-1)

                # pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
                
                # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
                # pred_ids = torch.where(pred_ids==-1, 0, pred_ids)

                # pred_motions = vq_model.forward_decoder(pred_ids)

                # pred_motions = vq_model.decoder(codes)
                # pred_motions = vq_model.forward_decoder(mids)

                et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len, pred_pose_eval.clone(),
                                                                  m_length)
                # em_pred = em_pred.unsqueeze(1)  #(bs, 1, d)
                motion_multimodality_batch.append(em_pred.unsqueeze(1))
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, 30, d)
            motion_multimodality.append(motion_multimodality_batch)
        else:
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            for k in range(bs):
                # index_motion = trans.sample(clip_text[k:k+1], False)
                try:
                    index_motion = res_model.sample(clip_text[k:k+1], False)
                    pred_ids = trans.generate(clip_text[k:k+1], index_motion, m_length // 4, time_steps, cond_scale,
                        temperature=temperature, topk_filter_thres=topkr,
                        gsample=gsample, force_mask=force_mask)
                    # index_motion = trans.sample(feat_clip_text[k:k+1], False)
                except:
                    index_motion = torch.ones(1,1).cuda().long()
                # RVQ
                # index_motion.unsqueeze_(-1)
                pred_pose = vq_model.forward_decoder(pred_ids)
                cur_len = pred_pose.shape[1]
                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            # mids = trans.generate(clip_text, m_length // 4, time_steps, cond_scale,
            #                       temperature=temperature, topk_filter_thres=topkr,
            #                       force_mask=force_mask)

            # motion_codes = motion_codes.permute(0, 2, 1)
            # mids.unsqueeze_(-1)
            # pred_ids = res_model.generate(mids, clip_text, m_length // 4, temperature=1, cond_scale=res_cond_scale)
            # pred_codes = trans(code_indices[..., 0], clip_text, m_length//4, force_mask=force_mask)
            # pred_ids = torch.where(pred_ids == -1, 0, pred_ids)

            # pred_motions = vq_model.forward_decoder(pred_ids)
            # pred_motions = vq_model.forward_decoder(mids)

            # et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len,
            #                                                   pred_motions.clone(),
            #                                                   m_length)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len,
                                                    pred_pose_eval.clone(),
                                                    m_length)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        # print(et_pred.shape, em_pred.shape)
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    if not force_mask and cal_mm:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Repeat {repeat_id} :, FID. {fid:.4f}, " \
          f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, " \
          f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, " \
          f"matching_score_real. {matching_score_real:.4f}, matching_score_pred. {matching_score_pred:.4f}," \
          f"multimodality. {multimodality:.4f}"
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, multimodality

@torch.no_grad()
def evaluation_qformer_retrieval(out_dir, val_loader, val_dataset, ep, eval_wrapper,
                           save_ckpt=False, save_anim=False):


    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    time_steps = 18
    if "kit" in out_dir:
        cond_scale = 2
    else:
        cond_scale = 4

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    correct = 0
    # for i in range(1):
    for batch in val_loader:
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        m_length = m_length.cuda()

        bs, seq = pose.shape[:2]
        # num_joints = 21 if pose.shape[-1] == 251 else 22

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(clip_text, word_embeddings, pos_one_hots, sent_len, pose, m_length)
        # visualize feature alignment heatmap
        similarity_matrix = torch.mm(et, em.t())
        similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
        plt.figure(figsize=(20, 20))
        sns.heatmap(similarity_matrix.cpu(), annot=False, cmap='YlOrRd', square=True, cbar_kws={'shrink': 0.8})
        plt.axis('off')
        cbar = plt.gca().collections[0].colorbar
        cbar.ax.tick_params(labelsize=30)  # 调整颜色条刻度的字号大小
        plt.savefig("heatmap.jpg")
        motion_annotation_list.append(em)

        # data = val_dataset.inv_transform(pose.detach().cpu().numpy())
        # pick_data = [torch.tensor(data[16]).unsqueeze(0), torch.tensor(data[3]).unsqueeze(0), torch.tensor(data[17]).unsqueeze(0), torch.tensor(data[9]).unsqueeze(0)]
        # data = torch.cat(pick_data, dim=0)
        # data = data.detach().cpu().numpy()
        # captions = [clip_text[16], clip_text[3], clip_text[17], clip_text[9]]
        # # print(ep_curves.shape)
        # save_dir = os.path.join('./demo', 'animation_for_check', 'E%04d' % 3)
        # os.makedirs(save_dir, exist_ok=True)
        # for i, (caption, joint_data) in enumerate(zip(captions, data)):
        #     joint_data = joint_data[:196]
        #     joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
        #     save_path = pjoin(save_dir, '%02d.mp4'%i)
        #     plot_3d_motion(save_path, t2m_kinematic_chain, joint, title=caption, fps=20)


        # temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=10, sum_all=True)
        # temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        temp_R = calculate_R_precision(em.cpu().numpy(), et.cpu().numpy(), top_k=10, sum_all=True)
        temp_match = euclidean_distance_matrix(em.cpu().numpy(), et.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample


    msg = f"--> \t Eva. Ep {ep} :Diversity Real. {diversity_real:.4f},  R_precision_real. {R_precision_real}, matching_score_real. {matching_score_real}"
    print(msg)


    return diversity_real, R_precision_real, matching_score_real

@torch.no_grad()
def generate_demo(val_loader, vq_model, trans, eval_wrapper,
                                time_steps, cond_scale, temperature, topkr, gsample=True, force_mask=False, cal_mm=True):
    trans.eval()
    vq_model.eval()

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    multimodality = 0

    nb_sample = 0
    if cal_mm:
        num_mm_batch = 3
    else:
        num_mm_batch = 0
    seq_len = torch.tensor(49).to('cuda:0')
    clip_text = 'A person is listening to someone speak from behind and occasionally nodding.'
    pred_ids = trans.generate(clip_text, seq_len, time_steps, cond_scale,
            temperature=temperature, topk_filter_thres=topkr,
            gsample=gsample, force_mask=force_mask)

    pred_pose = vq_model.forward_decoder(pred_ids[0][:seq_len[0]])
    data = pred_pose.detach().cpu().numpy()
    captions = [clip_text]
    lengths = seq_len[0].cpu().numpy()
    save_dir = os.path.join('./demo', 'animation', 'E%04d' % 3)
    os.makedirs(save_dir, exist_ok=True)
    # print(lengths)
    data = val_loader.inv_transform(data)
    # print(ep_curves.shape)
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:seq_len[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
        save_path = pjoin(save_dir, '%02d.mp4'%i)
        plot_3d_motion(save_path, t2m_kinematic_chain, joint, title=caption, fps=20)
    # return fid, diversity, R_precision, matching_score_pred, multimodality