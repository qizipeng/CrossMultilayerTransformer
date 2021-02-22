from Networks import Network
import os
import numpy as np
import torch.optim as optim
import time
import cv2
from tensorboardX import SummaryWriter
import torch
from utils import cal_mIoU_F1score,LRSchedulerHead,print_infos,save_seg_output_infos,compute_loss
from PIL import Image

class Segmentor(object):
    def __init__(self, args,config,data_loaders):
        self.paras = {
            'prior_mean': config['PRIOR_MEAN'],
            'prior_std': config['PRIOR_STD'],
            'log_path': args.log_path,
            'model_path': args.model_path,
            'data_loaders': data_loaders,
            'is_val': args.is_val,
            'train_batch_size': args.batch_size,
            'val_batch_size': args.val_batch_size,
            'device': torch.device('cpu') if (len(config['GPU_IDS']) == 0 or not torch.cuda.is_available()) else torch.device(f'cuda:%d' % config['GPU_IDS'][0]),
            'running_acc': {'loss': [], 'miou': [], 'f1_score': []},
            'epoch_metrics': {'loss': 1e10, 'miou': 0, 'f1_score': 0},
            'epoch_to_start': 0,
            'best_val_metrics': {'epoch_id': 0, 'loss': 1e10, 'miou': 0, 'f1_score': 0},
            'max_epoch': args.max_epoch,
            'print_info_interval': args.print_info_interval,
            'save_img_interval': args.save_img_interval,
            'writer': SummaryWriter(args.log_path + '/logs')
        }
        os.makedirs(self.paras['model_path'], exist_ok=True)
        os.makedirs(self.paras['log_path'] + '/train', exist_ok=True)
        if args.is_val:
            os.makedirs(self.paras['log_path'] + '/val', exist_ok=True)

        self.model = Network(in_channel=3, model_name=args.Network, args = args, gpu_ids=config['GPU_IDS']).model
        params_list = [{'params': self.model.parameters(), 'lr': args.lr}]
        if hasattr(self.model, '_flow_model'):
            params_list.append({'params': self.model.parameters(), 'lr': args.lr * 10})

        if args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(params_list)
        elif args.optimizer == 'Adam':
            self.optimizer = optim.Adam(params_list)
        else:
            self.optimizer = optim.SGD(params_list, momentum=0.9, nesterov=True)

        self._load_checkpoint(config['RESUME_WEIGHTS_PATH'])
        self.lr_scheduler = LRSchedulerHead(args.lr_scheduler_mode, args.lr, args.max_epoch, len(data_loaders['train']))


    def _load_checkpoint(self, weights_path):
        if os.path.exists(weights_path):
            print('loading model from pre-trained checkpoint %s' % os.path.basename(weights_path), flush=True)
            # load the entire checkpoint
            checkpoint = torch.load(weights_path, map_location=self.paras['device'])

            # update states
            # from collections import OrderedDict
            # new_dict = OrderedDict()
            # for key, value in checkpoint['model_state_dict'].items():
            #     key = key.replace('module.', '')
            #     new_dict[key] = value

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # update some other states
            self.paras['epoch_to_start'] = checkpoint['epoch_id'] + 1
            self.paras['best_val_metrics'] = checkpoint['best_val_metrics']

            print('Epoch_to_start = %d, Historical_best_val_metric = %.4f (at epoch %d)' %
                  (self.paras['epoch_to_start'], self.paras['best_val_metrics']['miou'],
                   self.paras['best_val_metrics']['epoch_id']), flush=True)
        else:
            print('training model from scratch...', flush=True)

    def _save_checkpoint(self, ckpt_name, epoch_id):
        torch.save({
            'epoch_id': epoch_id,
            'best_val_metrics': self.paras['best_val_metrics'],
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(self.paras['model_path'], ckpt_name))

    def _update_lr_schedulers(self, batch_id=0, epoch_id=0):
        self.lr_scheduler(self.optimizer, batch_id, epoch_id, self.paras['best_val_metrics']['miou'])

    def _collect_running_batch_states(self, pattern, epoch_id, batch_id, output, batch, loss, lr, fps_data_load, fps):

        if np.mod(batch_id, self.paras['print_info_interval']) == 1:
            miou, f1_score = cal_mIoU_F1score(output, batch['label'].to(self.paras['device']))

            self.paras['running_acc']['loss'].append(loss.detach().cpu().numpy())
            self.paras['running_acc']['miou'].append(miou[1].detach().cpu().numpy())
            self.paras['running_acc']['f1_score'].append(f1_score[1].detach().cpu().numpy())

            infos = {
                'pattern': pattern,
                'epoch_id': epoch_id,
                'max_epoch': self.paras['max_epoch'] - 1,
                'batch_id': batch_id,
                'batch_num': len(self.paras['data_loaders'][pattern]),
                'lr': lr,
                'fps_data_load': fps_data_load,
                'fps': fps,
                'loss': self.paras['running_acc']['loss'][-1],
                'miou': self.paras['running_acc']['miou'][-1],
                'f1_score': self.paras['running_acc']['f1_score'][-1]

            }
            print_infos(self.paras['writer'], infos)

        if np.mod(batch_id, self.paras['save_img_interval']) == 1:
            save_seg_output_infos(batch, output, self.paras['log_path'],
                                  pattern, epoch_id, batch_id, self.paras['prior_mean'], self.paras['prior_std'])

    def _collect_epoch_states(self, pattern, epoch_id):
        for key, value in self.paras['running_acc'].items():
            self.paras['epoch_metrics'][key] = np.mean(value)

        keys = self.paras['epoch_metrics'].keys()
        values = self.paras['epoch_metrics'].values()
        infos_str = 'Pattern: %s Epoch %d / %d' % (pattern, epoch_id, self.paras['max_epoch'] - 1)
        extra_infos = [f', {x}: {y:.4f}' for x, y in zip(keys, values)]
        infos_str = infos_str + ''.join(extra_infos)
        print(infos_str, flush=True)

        for key, value in zip(keys, values):
            self.paras['writer'].add_scalar(f'%s/epoch/%s' % (pattern, key), value, epoch_id)

    def _update_checkpoints(self, epoch_id):
        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt', epoch_id=epoch_id)

        print("Lastest model updated!", flush=True)
        keys = list(self.paras['epoch_metrics'].keys())
        values = list(self.paras['epoch_metrics'].values())
        infos_str = "Current_Epoch:\t\t"
        extra_infos = [f' {x}: {y:.4f}' for x, y in zip(keys, values)]
        infos_str = infos_str + ''.join(extra_infos)
        print(infos_str, flush=True)

        keys = list(self.paras['best_val_metrics'].keys())
        values = list(self.paras['best_val_metrics'].values())
        infos_str = "Best_Epoch:"
        extra_infos = [f' {x}: {y:.4f}' for x, y in zip(keys, values)]
        infos_str = infos_str + ''.join(extra_infos)
        print(infos_str, flush=True)
        print(flush=True)

        # update the best model (based on eval acc)
        if self.paras['epoch_metrics']['miou'] > self.paras['best_val_metrics']['miou']:
            self.paras['best_val_metrics']['epoch_id'] = epoch_id
            for key, value in self.paras['epoch_metrics'].items():
                self.paras['best_val_metrics'][key] = value
            self._save_checkpoint(ckpt_name='best_ckpt.pt', epoch_id=epoch_id)
            print('*' * 10 + 'Best model updated!' + '*' * 10, flush=True)
            print(flush=True)

    def _clear_cache(self):
        for key in self.paras['running_acc'].keys():
            self.paras['running_acc'][key] = []

    def _forward(self, batch, with_loss=True):
        img = batch['img'].to(self.paras['device'])
        label = batch['label'].to(self.paras['device'])
        # forward
        logits = self.model(img)
        loss = 0
        if with_loss:
            loss = compute_loss(logits, label, 'bce')
            # loss2 = compute_loss(out_second,label,'bce')
            # loss = loss1+loss2
        return logits, loss

    def train_model(self):
        for epoch_id in range(self.paras['epoch_to_start'], self.paras['max_epoch']):
            self._clear_cache()
            pattern = 'train'
            self.model.train()  # Set model to training mode
            time_2 = -1
            for batch_id, batch in enumerate(self.paras['data_loaders']['train']):
                time_1 = time.time()
                if time_2 != -1:
                    fps_data_load = len(batch['img']) / (time_1 - time_2)
                else:
                    fps_data_load = 0
                output, loss = self._forward(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._update_lr_schedulers(batch_id=batch_id, epoch_id=epoch_id)
                time_2 = time.time()
                fps = len(batch['img']) / (time_2 - time_1)
                lr = self.optimizer.param_groups[0]['lr']
                self._collect_running_batch_states(pattern, epoch_id, batch_id, output, batch, loss, lr, fps_data_load,
                                                   fps)
            self._collect_epoch_states(pattern, epoch_id)

            if self.paras['is_val']:
                print('Begin evaluation...', flush=True)
                self._clear_cache()
                pattern = 'val'
                self.model.eval()
                time_2 = -1
                for batch_id, batch in enumerate(self.paras['data_loaders']['val'], 0):
                    time_1 = time.time()
                    if time_2 != -1:
                        fps_data_load = len(batch['img']) / (time_1 - time_2)
                    else:
                        fps_data_load = 0
                    with torch.no_grad():
                        output, loss = self._forward(batch)
                    time_2 = time.time()
                    fps = len(batch['img']) / (time_2 - time_1)
                    lr = self.optimizer.param_groups[0]['lr']
                    self._collect_running_batch_states(pattern, epoch_id, batch_id, output, batch, loss, lr,
                                                       fps_data_load, fps)
                self._collect_epoch_states(pattern, epoch_id)
            self._update_checkpoints(epoch_id)
            if epoch_id % 20 == 0:
                self._save_checkpoint(ckpt_name='%d_ckpt.pt' % epoch_id, epoch_id=epoch_id)

    def _collect_testing_info(self, batch, output):
        out = postprocess(output).cpu()
        for img_id in range(len(out)):
            file_name = self.test_path + '/' + os.path.basename(batch['img_names'][img_id]).replace('.tif', '.png')
            pred = cv2.resize(out[img_id].numpy().astype('uint16'), dsize=eval(self.config['CROP_IMG_SIZE']),
                              interpolation=cv2.INTER_NEAREST)
            pred = (pred + 1) * 100
            img = Image.fromarray(pred)
            img.save(file_name)