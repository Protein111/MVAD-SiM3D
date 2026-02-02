import os
import re
import shutil
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

# try:
#     from apex import amp
#     from apex.parallel import DistributedDataParallel as ApexDDP
#     from apex.parallel import convert_syncbn_model as ApexSyncBN
# except:
#     from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN

from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp

@TRAINER.register_module
class MVADTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(MVADTrainer, self).__init__(cfg)

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        if self.imgs.dim() == 5:
            batch_size, view_count = self.imgs.shape[:2]
            self.imgs = self.imgs.view(batch_size * view_count, *self.imgs.size()[2:])
            if self.imgs_mask.dim() == 5:
                self.imgs_mask = self.imgs_mask.view(batch_size * view_count, *self.imgs_mask.size()[2:])
        if isinstance(inputs['cls_name'][0], str):
            self.cls_name = list(inputs['cls_name'])
            self.anomaly = torch.tensor(list(inputs['anomaly']))
        else:
            self.cls_name = []
            for i in range(len(inputs['cls_name'][0])):
                for j in range(len(inputs['cls_name'])):
                    self.cls_name.append(inputs['cls_name'][0][i])
            anomaly = []
            for i in range(len(inputs['anomaly'][0])):
                for j in range(len(inputs['anomaly'])):
                    anomaly.append(inputs['anomaly'][j][i])
            self.anomaly = torch.tensor(anomaly)
        self.sample_anomaly = inputs.get('sample_anomaly', self.anomaly)
        self.bs = self.imgs.shape[0]
        if isinstance(inputs['img_path'][0], str):
            self.imgs_path = list(inputs['img_path'])
        else:
            imgs_path = []
            for i in range(len(inputs['img_path'][0])):
                for j in range(len(inputs['img_path'])):
                    imgs_path.append(inputs['img_path'][j][i])
            self.imgs_path = imgs_path

    def forward(self):
        self.feats_t, self.feats_s = self.net(self.imgs)

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
        self.backward_term(loss_mse, self.optim)
        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
                        1,
                        self.master)

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        sample_view_counts = {}
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            # if batch_idx == 10:
            # 	break
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            # get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s,
                                                            [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False,
                                                            amap_mode='add', gaussian_sigma=4)
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            for idx, img_path in enumerate(self.imgs_path):
                if img_path is None:
                    continue
                full_img_path = os.path.join(self.cfg.data.root, str(img_path).lstrip('/'))
                sample_dir = os.path.dirname(os.path.dirname(full_img_path))
                mvad_dir = os.path.join(sample_dir, 'MVAD')
                os.makedirs(mvad_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(full_img_path))[0]
                parts = base_name.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    view_id = parts[0].zfill(2)
                else:
                    match = re.search(r'(\d+)$', base_name)
                    if match:
                        view_id = match.group(1).zfill(2)
                    else:
                        view_idx = sample_view_counts.get(sample_dir, 0) + 1
                        sample_view_counts[sample_dir] = view_idx
                        view_id = f"{view_idx:02d}"
                out_path = os.path.join(mvad_dir, f"{view_id}.npy")
                np.save(out_path, anomaly_map[idx])
            if self.cfg.vis and self.master:
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(self.imgs_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map,
                               self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            # ---------- log ----------
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
                    log_msg(self.logger, msg)
            