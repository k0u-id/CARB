# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import mmcv
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from mmseg.ops import resize
from ..builder import HEADS, build_head, build_backbone
from .decode_head import BaseDecodeHead
from mmcv.runner import force_fp32
from ..losses import accuracy


@HEADS.register_module()
class CARBHead(BaseDecodeHead):

    def __init__(self, decode_module_cfg, text_categories, text_channels, text_embeddings_path,
                    clip_unlabeled_cats=[], clip_cfg=None, clip_weights_path=None, clip_channels=None,
                    vit=False, adaptive=False, queue_size=5 ,coeff=1, warmup_iter=8000,
                    patch_size=(512, 256), resize_rate=1, resize_offset=1, dual_path=True,
                    get_train_mask=False, reset_counter=False, **kwargs):
        super(CARBHead, self).__init__(
            input_transform=decode_module_cfg.pop('input_transform'), **kwargs)
        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.clip_unlabeled_cats = torch.tensor(clip_unlabeled_cats, device='cuda')
        self.register_buffer('_iter_counter', torch.tensor(0, device='cuda'))
        self.clip_weights_path = clip_weights_path
        self.reset_counter = reset_counter
        self.warmup_iter = warmup_iter

        if clip_channels is None:
            clip_channels = self.in_channels
        
        self.adaptive = adaptive
        self.queue_size = queue_size
        if self.adaptive:
            self.coeff = coeff
            self.and_losses = torch.zeros(queue_size, device='cuda')
            self.vit_losses = torch.zeros(queue_size, device='cuda')

        self.coeff_loss = coeff
        self.patch_size = patch_size
        self.resize_rate = resize_rate
        self.resize_offset = resize_offset
        self.dual_path = dual_path
        self.get_train_mask = get_train_mask

        del self.conv_seg
        self.init_cfg = None

        decode_module_cfg.update(kwargs)
        self.build_decode_module(decode_module_cfg) # this case. it is ASPPHeadV2

        self.register_buffer('text_embeddings', torch.randn(text_categories, text_channels))

        self.vit = vit
        self.clip = build_backbone(clip_cfg)
        if self.vit:
            self.proj = nn.Conv2d(clip_channels, text_channels, 1, bias=False)
        else:
            self.q_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.k_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.v_proj = nn.Conv2d(clip_channels, clip_channels, 1)
            self.c_proj = nn.Conv2d(clip_channels, text_channels, 1)

    def init_weights(self, call_super=True):
        if call_super:
            super(CARBHead, self).init_weights()
        self.load_text_embeddings()
        self.load_clip_weights()

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cuda')
        self.text_embeddings[:, :] = loaded[:, :]
        print_log(f'Loaded text embeddings from {self.text_embeddings_path}', logger=get_root_logger())

    def load_clip_weights(self):
        loaded = torch.load(self.clip_weights_path, map_location='cuda')
        self.clip.load_state_dict(loaded['clip'])
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        for attr in attrs:
            current_attr = getattr(self, attr)
            state_dict = loaded[attr]
            for key in state_dict:
                if 'weight' in key:
                    state_dict[key] = state_dict[key][:, :, None, None]
            current_attr.load_state_dict(state_dict)
        print_log(f'Loaded clip weights from {self.clip_weights_path}', logger=get_root_logger())

    def _freeze(self):
        """Freeze params and norm stats."""
        super(CARBHead, self)._freeze()
        # always freeze these modules
        attrs = ['proj'] if self.vit else ['q_proj', 'k_proj', 'v_proj', 'c_proj']
        attrs.append('clip')
        for attr in attrs:
            i = getattr(self, attr)
            for m in i.modules():
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def build_decode_module(self, cfg):
        cfg['init_cfg'] = None
        cfg['in_channels'] = self.in_channels
        cfg['channels'] = self.channels
        self.decode_module = build_head(cfg)
        del self.decode_module.loss_decode
        del self.decode_module.conv_seg
        del self.decode_module.dropout

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = F.conv2d(feat, self.text_embeddings[:, :, None, None]) # compare text embedding with ASPP features        
        return output

    def forward(self, inputs):
        output = self.decode_module.forward_module(inputs)
        output = self.cls_seg(output)

        if self.reset_counter:
            self.reset_counter = False
            self._iter_counter *= 0

        if self.training:
            self._iter_counter += 1
            if self._iter_counter == 0:
                print_log('Start clip guided training', logger=get_root_logger())
            if self._iter_counter == self.warmup_iter:
                print_log('End warm-up training', logger=get_root_logger())
                
        return output


    def gen_clip_mask(self, img, shape, img_label, unlabeled_cats=None):
         
        x = self.clip(img)[-1]
        
        q, k, v, cls_token = None, None, None, None
        if isinstance(x, list) and len(x) == 4:
            x, q, k, v = x      # assign vit output to each variable
        if isinstance(x, list) and len(x) == 2: # len(x) == 4
            x, cls_token = x    # assign x and cls_token 
        if v is not None:
            feat = self.proj(v) # if vit, project v to make feature
            # print(feat.shape) [4, 512, 32, 64]
        else:
            feat = self.proj(x) 
        if cls_token is not None:
            cls_token = self.proj(cls_token[:, :, None, None])[:, :, 0, 0]
        
        feat = feat / feat.norm(dim=1, keepdim=True) # normalize is true
        clip_semantic_seg = torch.zeros(img.shape[0], shape[0], shape[1], dtype=torch.int64).cuda() # [4, 512, 1024]

        text_embeddings = self.text_embeddings
        unlabeled_text = text_embeddings[unlabeled_cats]
        unlabeled_idx = (clip_semantic_seg == 0)

        output = torch.einsum('nchw,lc->nlhw', [feat, unlabeled_text])
        output = output * img_label.unsqueeze(-1).unsqueeze(-1)

        output = resize(
            input=output,
            size=clip_semantic_seg.shape[1:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        output = output.permute(0, 2, 3, 1)    
        match_matrix = output[unlabeled_idx]
        
        clip_semantic_seg[unlabeled_idx] = unlabeled_cats[match_matrix.argmax(dim=1)]
        clip_semantic_seg = clip_semantic_seg[:, None, :, :]
        clip_semantic_seg[clip_semantic_seg<0] = 255

        return clip_semantic_seg 

    def label_sanity_check(self, gt_semantic_seg):
        for i in self.clip_unlabeled_cats: # if label is not within 0~19, gt_semantic_seg will be True --> alarm
            assert torch.all(gt_semantic_seg != i), f'Ground-truth leakage! {i}'
    
    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, img=None, img_label=None):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        seg_logits = self.forward(inputs)
        gt_clip = None
        
        if self.dual_path:
            with torch.no_grad():
                img_shape = img.shape[2:]
                gt_clip = self.gen_clip_mask(img, img_shape, img_label, self.clip_unlabeled_cats)
        else:
            gt_clip = gt_semantic_seg

        # gt_clip : ViT, gt_semantic_seg : RN50, gt_self : self
        losses = self.losses(seg_logits, gt_clip, img, img_label)
        
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        output = self.forward(inputs)

        if self.get_train_mask:
            img_label = torch.tensor(img_metas[0]['img_label'], device=output.device)
            output = output * img_label.reshape(1, -1, 1, 1)
        
        return output

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label, img, img_label):
        """Compute segmentation loss."""
        loss = dict()
        
        seg_logit = resize(
            input=seg_logit,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)   # vit
        
        with torch.no_grad():

            croph = self.patch_size[0]
            cropw = self.patch_size[1]
            
            if img.shape[2] == croph:
                hpos = 0
            else:
                hpos = torch.randint(low=0, high=img.shape[2] - croph, size=())
            if img.shape[3] == cropw:
                wpos = 0
            else:
                wpos = torch.randint(low=0, high=img.shape[3] - cropw, size=())

            multiplier = self.resize_rate * torch.rand(size=()) + self.resize_offset

            partial_img = img[:, :, hpos:hpos+croph, wpos:wpos+cropw].clone()
            partial_img = resize(
                input=partial_img,
                size=(int(croph * multiplier), int(cropw * multiplier)),
                mode='bilinear'
            )      

            partial_label = self.gen_clip_mask(partial_img, (croph, cropw), img_label, self.clip_unlabeled_cats)
            partial_label = partial_label.squeeze(1)

        partial_logit = seg_logit[:, :, hpos:hpos+croph, wpos:wpos+cropw].clone()

        if self.warmup_iter is not None and self._iter_counter > self.warmup_iter:

            with torch.no_grad():

                gt_self = seg_logit * img_label.unsqueeze(-1).unsqueeze(-1)
                gt_self = gt_self.argmax(dim=1)
                gt_self = gt_self.squeeze(1)

                label_and = seg_label.clone()
                label_vit = seg_label.clone()

                label_and[seg_label != gt_self] = 255 # && area
                label_vit[seg_label == gt_self] = 255 # !& area

                partial_and = partial_label.detach().clone()
                partial_and[partial_and != gt_self[:, hpos:hpos+croph, wpos:wpos+cropw]] = 255

                partial_dif = partial_label.detach().clone()
                partial_dif[partial_and != 255] = 255

            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
            for loss_decode in losses_decode:

                loss_and = loss_decode(seg_logit, label_and, ignore_index=255)
                loss_vit = loss_decode(seg_logit, label_vit, ignore_index=255)
                loss_partand = loss_decode(partial_logit, partial_and, ignore_index=255)
                loss_partdif = loss_decode(partial_logit, partial_dif, ignore_index=255)
                
                with torch.no_grad():
                    if self.adaptive:
                        self.and_losses[self._iter_counter % self.queue_size] = loss_and
                        self.vit_losses[self._iter_counter % self.queue_size] = loss_vit

                        self.coeff_loss =  self.coeff * self.and_losses.mean() / self.vit_losses.mean()
                
                loss_temp = loss_and + loss_vit * self.coeff_loss + \
                            loss_partand + loss_partdif * self.coeff_loss
                
                if self._iter_counter % 50 == 0:
                    print("loss_and : %f, loss_vit : %f, loss_part : %f" % (loss_and, loss_vit * self.coeff_loss, loss_partdif * self.coeff_loss))
                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_temp
                else:
                    loss[loss_decode.loss_name] += loss_temp
        else:
            if not isinstance(self.loss_decode, nn.ModuleList):
                losses_decode = [self.loss_decode]
            else:
                losses_decode = self.loss_decode
            for loss_decode in losses_decode:

                loss_all = loss_decode(seg_logit, seg_label, ignore_index=255)
                loss_part = loss_decode(partial_logit, partial_label, ignore_index=255)
                loss_temp = loss_all + loss_part

                if loss_decode.loss_name not in loss:
                    loss[loss_decode.loss_name] = loss_temp
                else:
                    loss[loss_decode.loss_name] += loss_temp

        loss['acc_seg'] = accuracy(seg_logit, seg_label)
        return loss