import yaml
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
from typing import Optional, List
from torch import Tensor
import torch.nn.functional as F

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

class ParamsParser:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class LRScheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0, quiet=False):
        self.mode = mode
        self.quiet = quiet
        if not quiet:
            print('Using {} LR scheduler with warm-up epochs of {}!'.format(self.mode, warmup_epochs))
        if mode == 'step':
            assert lr_step
        self.base_lr = base_lr
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.total_iters = (num_epochs - warmup_epochs) * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = self.base_lr * 1.0 * T / self.warmup_iters
        elif self.mode == 'cos':
            T = T - self.warmup_iters
            lr = 0.5 * self.base_lr * (1 + math.cos(1.0 * T / self.total_iters * math.pi))
        elif self.mode == 'poly':
            T = T - self.warmup_iters
            lr = self.base_lr * pow((1 - 1.0 * T / self.total_iters), 0.9)
        elif self.mode == 'step':
            lr = self.base_lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        if epoch > self.epoch and (epoch == 0 or best_pred > 0.0):
            if not self.quiet:
                print('\n=>Epoch %i, learning rate = %.4f, \
                    previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        if lr < 1e-8:
            lr = 1e-8
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr


class LRSchedulerHead(LRScheduler):
    """Incease the additional head LR to be 10 times"""
    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10

def make_numpy_img(tensor_data):
    if len(tensor_data.shape) == 2:
        tensor_data = tensor_data.unsqueeze(2)
        tensor_data = torch.cat((tensor_data, tensor_data, tensor_data), dim=2)
    elif tensor_data.size(0) == 1:
        tensor_data = tensor_data.permute((1, 2, 0))
        tensor_data = torch.cat((tensor_data, tensor_data, tensor_data), dim=2)
    elif tensor_data.size(0) == 3:
        tensor_data = tensor_data.permute((1, 2, 0))
    elif tensor_data.size(2) == 3:
        pass
    else:
        raise Exception('tensor_data apply to make_numpy_img error')
    vis_img = tensor_data.detach().cpu().numpy()

    return vis_img

def inv_normalize_img(img, prior_mean=[0, 0, 0], prior_std=[1, 1, 1]):
    prior_mean = torch.tensor(prior_mean, dtype=torch.float).to(img.device).view(img.size(0), 1, 1)
    prior_std = torch.tensor(prior_std, dtype=torch.float).to(img.device).view(img.size(0), 1, 1)
    img = img * prior_std + prior_mean
    img = img * 255.
    img = torch.clamp(img, min=0, max=255)
    return img

def encode_onehot_to_mask(onehot):
    '''
    onehot: tensor, BxnxWxH or nxWxH
    output: tensor, BxWxH or WxH
    '''
    assert len(onehot.shape) in [3, 4], "encode_onehot_to_mask error!"
    mask = torch.argmax(onehot, dim=len(onehot.shape)-3)
    return mask

def save_seg_output_infos(input, output, vis_dir, pattern, epoch_id, batch_id, prior_mean, prior_std):
    pred_label = torch.argmax(output, 1)
    k = np.clip(int(0.2 * len(pred_label)), a_min=2, a_max=len(pred_label[0]))
    k=1
    ids = np.random.choice(range(len(pred_label)), k, replace=False)
    for img_id in ids:
        img = input['img'][img_id].to(pred_label.device)
        target = input['label'][img_id].to(pred_label.device)

        img = make_numpy_img(inv_normalize_img(img, prior_mean, prior_std)) / 255.
        target = make_numpy_img(encode_onehot_to_mask(target))
        pred = make_numpy_img(pred_label[img_id])

        vis = np.concatenate([img, pred, target], axis=0)
        vis = np.clip(vis, a_min=0, a_max=1)
        file_name = os.path.join(vis_dir, pattern, str(epoch_id) + '_' + str(batch_id) + '.jpg')
        plt.imsave(file_name, vis)

def print_infos(writer, infos: dict):
    keys = list(infos.keys())
    values = list(infos.values())
    infos_str = 'Pattern: %s [%d,%d][%d,%d], lr: %5f, fps_data_load: %.2f, fps: %.2f' % tuple(values[:8])
    if len(values) > 8:
        extra_infos = [f', {x}: {y:.4f}' for x, y in zip(keys[8:], values[8:])]
        infos_str = infos_str + ''.join(extra_infos)
    print(infos_str, flush=True)

    writer.add_scalar('%s/lr' % infos['pattern'], infos['lr'],
                      infos['epoch_id'] * infos['batch_num'] + infos['batch_id'])
    for key, value in zip(keys[8:], values[8:]):
        writer.add_scalar(f'%s/%s' % (infos['pattern'], key), value,
                          infos['epoch_id'] * infos['batch_num'] + infos['batch_id'])

def compute_loss(input, target, loss_name='cross_entropy'):
    if loss_name == "cross_entropy":
        loss = F.cross_entropy(input, target)
    elif loss_name == 'bce':
        loss = F.binary_cross_entropy_with_logits(input, target)
    return loss

def decode_mask_to_onehot(mask, n_class):
    '''
    mask : BxWxH or WxH
    n_class : n
    return : BxnxWxH or nxWxH
    '''
    assert len(mask.shape) in [2, 3], "decode_mask_to_onehot error!"
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0)
    onehot = torch.zeros((mask.size(0), n_class, mask.size(1), mask.size(2))).to(mask.device)
    for i in range(n_class):
        onehot[:, i, ...] = mask == i
    if len(mask.shape) == 2:
        onehot = onehot.squeeze(0)
    return onehot

def cal_mIoU_F1score(pred, target):
    '''
    pred: logits, tensor, nBatch*nClass*W*H
    target: labels, tensor, nBatch*nClass*W*H
    '''
    pred = torch.argmax(pred.detach(), dim=1)
    pred = decode_mask_to_onehot(pred, target.size(1))
    # positive samples in ground truth
    gt_pos_sum = torch.sum(target == 1, dim=(0, 2, 3))
    # positive prediction in predict mask
    pred_pos_sum = torch.sum(pred == 1, dim=(0, 2, 3))
    # cal true positive sample
    true_pos_sum = torch.sum((target == 1) * (pred == 1), dim=(0, 2, 3))
    # Precision
    precision = true_pos_sum / (pred_pos_sum + 1e-15)
    # Recall
    recall = true_pos_sum / (gt_pos_sum + 1e-15)
    # IoU
    iou = true_pos_sum / (pred_pos_sum + gt_pos_sum - true_pos_sum + 1e-15)
    # F1-score
    f1_score = 2 * precision * recall / (precision + recall + 1e-15)
    return iou, f1_score