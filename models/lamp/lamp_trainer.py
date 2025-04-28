import torch
from collections import defaultdict, OrderedDict
import torch.optim as optim
from models.lamp.optim import LinearWarmupCosineLRScheduler
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *
from torch.cuda.amp import autocast
import logging
from einops import rearrange, repeat
import torch.distributed as dist
from utils.metrics import *
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
class AverageMeter1(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        logging.info('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, correct

def test_zeroshot_3d_core(batch, motion_features, text_features, args=None, test_data=None):
    batch_time = AverageMeter1('Time', ':6.3f')
    top1 = AverageMeter1('Acc@1', ':6.2f') 
    top3 = AverageMeter1('Acc@3', ':6.2f')
    top5 = AverageMeter1('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(batch),
        [batch_time, top1, top3, top5],
        prefix='Test: ')
    logging.info('=> encoding captions')   
    motion_features = F.normalize(motion_features, dim=-1, p=2)
    text_features = F.normalize(text_features, dim=-1, p=2)
    end = time.time()
    per_class_stats = defaultdict(int)
    per_class_correct_top1 = defaultdict(int)
    per_class_correct_top3 = defaultdict(int)
    per_class_correct_top5 = defaultdict(int)
    logits_per_pc = motion_features.float() @ text_features.float().t()
    bs = motion_features.size(0)
    target = torch.linspace(0, bs - 1, bs, dtype=int).to(motion_features.device)
    # measure accuracy and record loss
    (acc1, acc3, acc5), correct = accuracy(logits_per_pc, target, topk=(1, 3, 5))
    # TODO: fix the all reduce for the correct variable, assuming only one process for evaluation!
    top1.update(acc1.item(), bs)
    top3.update(acc3.item(), bs)
    top5.update(acc5.item(), bs)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
    progress.synchronize()
    logging.info('0-shot * Acc@1 {top1.avg:.3f} Acc@3 {top3.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg, top3.avg, top5.avg


def def_value():
    return 0.0

class MotionQFormerTrainer:
    def __init__(self, args, motion_qformer, vq_model):
        self.opt = args
        self.motion_qformer = motion_qformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        self.scaler = torch.cuda.amp.GradScaler()
        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_qformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        # code_idx, _ = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # loss_dict = {}
        # self.pred_ids = []
        # self.acc = []

        _loss, text_feat, motion_feat = self.motion_qformer(motion, conds)
        # _loss, _pred_ids, _acc = self.motion_qformer(motion, conds)

        # return _loss.loss, _loss.loss_ptc, _loss.loss_ptm, _loss.loss_lm, text_feat, motion_feat
        return _loss.loss, _loss.loss_ptc, _loss.loss_ptm, _loss.loss_lm, _loss.loss_gen, text_feat, motion_feat
        # return _loss, _acc
    
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        p_wd, p_non_wd = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)        
        optim_params = [
            {"params": p_wd, "weight_decay": weight_decay, "lr_scale": lr_scale},
            {"params": p_non_wd, "weight_decay": 0, "lr_scale": lr_scale},
        ]                
        return optim_params
    # def update(self, batch_data):
        # loss, acc = self.forward(batch_data)
        # self.scheduler.step()
        # with autocast(enabled=True):
        # loss = self.forward(batch_data)
        # loss.backward()
        # self.scaler.step(self.opt_t2m_transformer)
        # self.scaler.update()
        # self.opt_qformer.zero_grad()
        # self.opt_qformer.step()

        # return loss.item()

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.motion_qformer.state_dict()
        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]
        state = {
            'motion_qformer': t2m_trans_state_dict,
            'opt_qformer': self.opt_qformer.state_dict(),
            # 'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['motion_qformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_qformer.load_state_dict(checkpoint['opt_qformer']) # Optimizer
            # self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.motion_qformer.to(self.device)
        self.vq_model.to(self.device)
        param = self.motion_qformer.get_optimizer_params(weight_decay=0.05, lr_scale=1)
        self.opt_qformer = torch.optim.AdamW(
                param,
                lr=float(1e-4),
                betas=(0.9, 0.99),
        )    
        self.scheduler = LinearWarmupCosineLRScheduler(
                        optimizer=self.opt_qformer,
                        max_epoch=self.opt.max_epoch,
                        min_lr=1e-5,
                        init_lr=1e-4,
                        decay_rate=1.,
                        # warmup_start_lr=1e-5,
                        warmup_start_lr=1e-6,
                        warmup_steps=2000,
        )
        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.motion_qformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                self.scheduler.step(cur_epoch=epoch, cur_step=it, max_warmupstep=self.opt.warm_up_iter)
                # if it < self.opt.warm_up_iter:
                #     self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                # # loss = self.update(batch_data=batch)
                self.opt_qformer.zero_grad()
                loss, loss_ptc, loss_ptm, loss_lm, loss_gen, _, _ = self.forward(batch_data=batch)
                # loss, loss_ptc, loss_ptm, loss_lm, _, _ = self.forward(batch_data=batch)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt_qformer)
                self.scaler.update()   
                logs['loss'] += loss.item()
                logs['loss_ptc'] += loss_ptc.item()
                logs['loss_ptm'] += loss_ptm.item()
                logs['loss_lm'] += loss_lm.item()
                logs['loss_gen'] += loss_gen.item()
                logs['lr'] += self.opt_qformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.motion_qformer.eval()

            val_loss = []
            val_lossptc = []
            val_lossptm = []
            val_losslm = []
            val_lossgen = []
            val_acc1 = 0.0
            val_acc2 = 0.0
            val_acc3 = 0.0
            match_score = 0.0
            nb_sample = 0
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    # loss, loss_ptc, loss_ptm, loss_lm, text_feat, motion_feat = self.forward(batch_data=batch)
                    loss, loss_ptc, loss_ptm, loss_lm, loss_gen, text_feat, motion_feat = self.forward(batch_data=batch)
                    bs = motion_feat.size(0)
                    nb_sample += bs
                    # acc1, acc3, acc5 = test_zeroshot_3d_core(batch, motion_feat, text_feat)
                    acc1, acc2, acc3 = calculate_R_precision(text_feat.cpu().detach().numpy(), motion_feat.cpu().detach().numpy(), top_k=3, sum_all=True)
                    temp_match = euclidean_distance_matrix(text_feat.cpu().detach().numpy(), motion_feat.cpu().detach().numpy()).trace()
                    val_loss.append(loss.item())
                    val_lossptc.append(loss_ptc.item())
                    val_lossptm.append(loss_ptm.item())
                    val_losslm.append(loss_lm.item())
                    val_lossgen.append(loss_gen.item())
                    val_acc1 += acc1
                    val_acc2 += acc2
                    val_acc3 += acc3
                    match_score += temp_match
            print(f"Validation loss:{np.mean(val_loss):.3f}, loss_ptc:{np.mean(val_lossptc):.3f}, loss_ptm:{np.mean(val_lossptm):.3f}," 
            f"loss_lm:{np.mean(val_losslm):.3f}, loss_gen:{np.mean(val_lossgen):.3f}, acc1:{val_acc1 / nb_sample:.3f}, acc2:{val_acc2 / nb_sample:.3f}, acc3:{val_acc3 / nb_sample:.3f}, match_score:{match_score / nb_sample:.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/loss_ptc', np.mean(val_lossptc), epoch)
            self.logger.add_scalar('Val/loss_ptm', np.mean(val_lossptm), epoch)
            self.logger.add_scalar('Val/loss_lm', np.mean(val_losslm), epoch)
            self.logger.add_scalar('Val/loss_gen', np.mean(val_lossgen), epoch)
            self.logger.add_scalar('Val/acc1', val_acc1 / nb_sample, epoch)
            self.logger.add_scalar('Val/acc2', val_acc2 / nb_sample, epoch)
            self.logger.add_scalar('Val/acc3', val_acc3 / nb_sample, epoch)
            self.logger.add_scalar('Val/match score', match_score / nb_sample, epoch)

            if (val_acc1 / nb_sample) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {val_acc1 / nb_sample}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = val_acc1 / nb_sample


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            # self.l1_criterion = torch.nn.SmoothL1Loss()


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):

        conds, motion, m_lens = batch_data
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, all_codes = self.vq_model.encode(motion)
        m_lens = m_lens // 4

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens)

        return ce_loss, acc

    def update(self, batch_data):
        loss, acc = self.forward(batch_data)

        self.opt_res_transformer.zero_grad()
        loss.backward()
        self.opt_res_transformer.step()
        self.scheduler.step()

        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
        for e in clip_weights:
            del res_trans_state_dict[e]
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_model.') for k in missing_keys])

        try:
            self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer']) # Optimizer

            self.scheduler.load_state_dict(checkpoint['scheduler']) # Scheduler
        except:
            print('Resume wo optimizer')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)

        self.opt_res_transformer = optim.AdamW(self.res_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')  # TODO
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
            self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_loss = 100
        best_acc = 0

        while epoch < self.opt.max_epoch:
            self.res_transformer.train()
            self.vq_model.eval()

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # self.logger.add_scalar('val_loss', val_loss, it)
                    # self.l
                    for tag, value in logs.items():
                        self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

            self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_loss) < best_loss:
                print(f"Improved loss from {best_loss:.02f} to {np.mean(val_loss)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_loss = np.mean(val_loss)

            if np.mean(val_acc) > best_acc:
                print(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                # self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
                self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                best_matching=best_matching, eval_wrapper=eval_wrapper,
                plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
            )