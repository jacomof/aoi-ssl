import random
import torch
import numpy as np
import torch.nn as nn

from segmentation.models.vit import init_vit
from segmentation.models.vit.downsample_vit import DownsampleVisionTransformer 
from segmentation.models.ema import ModelEma
from ..mae.mae import MaskedAutoencoder
from . import ibot_loss
from ..dino.dino_head import DINOHead
from torch import distributed as dist
import torch.nn.functional as F
#from segmentation.models.swinv2.swin_transformer_v2 import init_swin_transformer_v2_ssl
from segmentation.models.faster_vit.faster_vit_any_res import faster_vit_1_ssl_any_res, faster_vit_0_ssl_any_res

import math

from segmentation.models.vit.layers import (
    MemEffAttention,
)

from segmentation.models.vit.layers.block import DownsampleBlock

from functools import partial

def init_vit_tiny_downsample(size, num_register_tokens=0, **kwargs):
    model = DownsampleVisionTransformer(
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        block_fn=partial(DownsampleBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model

class iBot(MaskedAutoencoder):

    def __init__(self, **parameters):
        super().__init__(**parameters)

        self.patch_size = parameters.get("patch_size", 16)

        self.dino_loss_weight = parameters.get("dino_loss_weight", 1.0)
        self.reconstruction_loss_weight = parameters.get("reconstruction_loss_weight", 0.0)
        self.ncrops = parameters.get("nlocal_crops", 1) + 2 # We add two because there are always 2 global crops
        print("Number of crops: ", self.ncrops)
        self.normalize_loss = parameters.get("normalize_loss", False)
        self.centering_mode = parameters.get("centering_mode", None)
        self.use_downsample_vit = parameters.get("use_downsample_vit", False)
        self.faster_vit = parameters.get("faster_vit", False)
        self.student_temp = parameters.get("student_temp", 0.1)
        self.teacher_temp = parameters.get("teacher_temp", 0.04)
        self.center_momentum = parameters.get("center_momentum", 0.9)
        self.embs_before_head = parameters.get("embs_before_head", False)
        self.use_projector = parameters.get("use_projector", True)
        self.dino_out_dim = parameters.get("out_dim", 768)

        self.use_swin = parameters.get("use_swin", False)

        self.use_faster_vit_0 = parameters.get("use_fastervit_0", False)
        self.use_faster_vit_1 = parameters.get("use_fastervit_1", False)

        if self.use_downsample_vit:
            print("Initializing Downsample ViT")
            self.encoder = init_vit_tiny_downsample(
                parameters["vit_type"],
                num_register_tokens=self.num_reg_tokens,
                **parameters
            )
        # elif self.use_swin:
        #     print("Initializing Swin Transformer V2")
        #     self.encoder = init_swin_transformer_v2_ssl()
        elif self.use_faster_vit_0:
            print("Initializing Faster ViT 0")
            self.encoder = faster_vit_0_ssl_any_res()
        elif self.use_faster_vit_1:
            print("Initializing Faster ViT 1")
            self.encoder = faster_vit_1_ssl_any_res()
        else:
            print("Initializing ViT")
            self.encoder = init_vit(
                parameters["vit_type"],
                num_register_tokens=self.num_reg_tokens,
                **parameters
            )


        self.teacher_ema = ModelEma(
            self.encoder,
            decay=0.998,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        self.encoder_teacher = self.teacher_ema.module

        self.loss = ibot_loss.iBotLoss(
            student_temp_cls=parameters["student_temp_cls"],
            teacher_temp_cls=parameters["teacher_temp_cls"],
            center_momentum_cls=parameters["center_momentum_cls"],

            student_temp_patch=parameters["student_temp_patch"],
            teacher_temp_patch=parameters["teacher_temp_patch"],
            center_momentum_patch=parameters["center_momentum_patch"],

            out_dim=parameters["out_dim"],
            num_patches=self.encoder.num_patches, # is the number of patches of the global crops, not the entire image

            patch_size=self.patch_size,
        )

        self.student_dino_head = DINOHead(
            in_dim=self.encoder.embed_dim,
            out_dim=self.dino_out_dim,
        ) if self.use_projector else None

        self.teacher_dino_head = DINOHead(
            in_dim=self.encoder.embed_dim,
            out_dim=self.dino_out_dim,
        ) if self.use_projector else None

        if self.normalize_loss:
            print("Normalizing loss")
            self.register_buffer("dino_loss_running_max", torch.tensor(float("-inf")))
            self.register_buffer("dino_loss_running_min", torch.tensor(float("inf")))
            self.register_buffer("mae_loss_running_max", torch.tensor(float("-inf")))
            self.register_buffer("mae_loss_running_min", torch.tensor(float("inf")))


        # self.use_mae declared in the parent class, otherwise
        # the decoder would be always initialized 

    def forward_ibot(self, x, encoder, head, masks=None, compute_locals=True):
        """iBot forward pass.

        Args:
            x (torch.Tensor): List of input tensors with different crops, each of shape (B, C, Hi, Wi).
            encoder (nn.Module): Encoder model.
            head (nn.Module): Head model to apply on the features.

        Returns:
            outputs_after_head (torch.Tensor): CLS embedding after applying the DiNo head.
            global_patch_embs (list): List of patch embeddings for global crops.

        """

        "Input : [B x crop1, B x crop2, ...]"
        "Masks : [B x (L), B x (L), ...] where L is the sequence length of the crop"
        if not isinstance(x, list):
            x = [x]
        
        # Groups crops of the same size together
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0) # Assumes that tensors of the same size are contiguous in the list

        cls_embs = []
        global_patch_embs = []

        # First compute global crops
        start_idx = 0
        global_inp = torch.cat(x[start_idx:idx_crops[0]]) # Concatenate the remaining crops
        global_masks = torch.cat(masks[start_idx:idx_crops[0]]) if masks is not None else None
        global_embs = encoder(global_inp, global_masks) # (2*B)xD
        cls_embs.append(global_embs["x_norm_clstoken"]) # (2*B)xD
        global_patch_embs = global_embs["x_norm_patchtokens"].chunk(2, dim=0) # [(B, P, D), (B, P, D)]

        # print(f"Global patch embedding length: {len(global_patch_embs)}")
        # print(f"Global patch embedding shape: {global_patch_embs[0].shape}")


        if compute_locals:
            # Then compute local crops
            local_inp = torch.cat(x[idx_crops[0]:]) # Concatenate the local crops
            local_embs = encoder(local_inp) #BxD
            cls_embs.append(local_embs["x_norm_clstoken"])

        # Run the head forward on the concatenated features.
        cls_embs = torch.cat(cls_embs) # (B*ncrops)xD

        batch_size = x[0].shape[0]
        hidden_dim = self.dino_out_dim

        # print(f"Global patch embedding list length: {len(global_patch_embs)}")
        # print(f"CLS embedding shape: {cls_embs.shape}")
        patch_embs_view_1 = global_patch_embs[0].flatten(0, 1) # (B*P)xD
        patch_embs_view_1_after_head = head(patch_embs_view_1) # (B*P)xD
        patch_embs_view_1_tokenized = patch_embs_view_1_after_head.reshape(batch_size, -1, hidden_dim) # (B, P, D)

        patch_embs_view_2 = global_patch_embs[1].flatten(0, 1) # (B*P)xD
        patch_embs_view_2_after_head = head(patch_embs_view_2) # (B*P)xD
        patch_embs_view_2_tokenized = patch_embs_view_2_after_head.reshape(batch_size, -1, self.dino_out_dim) # (B, P, D)

        cls_outputs_after_head: torch.Tensor = head(cls_embs)

        return cls_outputs_after_head, patch_embs_view_1_tokenized, patch_embs_view_2_tokenized, cls_embs

    def update_running_min_max(self, loss: torch.Tensor, loss_min: torch.Tensor, loss_max: torch.Tensor):
        with torch.no_grad():
            loss_max.copy_(torch.max(loss_max, loss.detach()))
            loss_min.copy_(torch.min(loss_min, loss.detach()))


    def forward(self, x: list[torch.Tensor], masks: list[torch.Tensor]):
        """ Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            mask_ratio (float): Ratio of the input to be masked.
        
        Returns:
            loss (torch.Tensor): Loss tensor.
            embs_student (torch.Tensor): Student cls embeddings.
            embs_teacher (torch.Tensor): Teacher cls embeddings.
        
        """
        # Get patches for encoder
        # x_1 = x[:,0,:,:].unsqueeze(1)
        # x_2 = x[:,1,:,:].unsqueeze(1)
        # x = torch.cat((x_1, x_2), dim=0)

        # print("Length of input shape: ", len(x))

        # for crop in x:
        #     print(f"Crop shape: {crop.shape}")

        # for mask in masks:
        #     print(f"Mask shape: {mask.shape}")

        loss = None
        embs_student = None
        embs_teacher = None
        
        x_globals = x[:2]

        student_cls_after_head, student_patch_embs_view_1_tokenized, student_patch_embs_view_2_tokenized, \
             student_cls_before_head = self.forward_ibot(x, self.encoder, self.student_dino_head, masks, \
                                                        compute_locals=True)
        student_cls_list = student_cls_after_head.chunk(self.ncrops) # nlocal_crops + 2
        student_cls_before_head_mean = student_cls_before_head.detach().mean() # [BxD, BxD] -> [BxD]
        student_embs_avg_for_log = ((student_cls_list[0] + student_cls_list[1]) / 2.0).detach()
        student_embs_avg_for_log = F.softmax(student_embs_avg_for_log / self.student_temp, dim=-1)

        teacher_cls, teacher_patch_embs_view_1_tokenized, teacher_patch_embs_view_2_tokenized, _ =\
            self.forward_ibot(x_globals, self.encoder_teacher, self.teacher_dino_head, masks=None, compute_locals=False)

        teacher_cls_centered_sharpened = self.loss.softmax_center_teacher_cls(teacher_cls)
        teacher_cls_list = teacher_cls_centered_sharpened.detach().chunk(2)
        teacher_embs = (teacher_cls_list[0] + teacher_cls_list[1])/2.0  # [BxD, BxD] -> [BxD]



        dino_loss = self.loss.dino_loss(student_cls_list, teacher_cls_list)

        teacher_patch_embs_view_1_tokenized = teacher_patch_embs_view_1_tokenized
        teacher_patch_embs_view_2_tokenized = teacher_patch_embs_view_2_tokenized
        teacher_patch_output = torch.cat(
            [teacher_patch_embs_view_1_tokenized, teacher_patch_embs_view_2_tokenized], dim=0
        )
        teacher_patch_centered_sharpened = self.loss.softmax_center_teacher_patch(teacher_patch_output).detach().chunk(2)
        teacher_patch_embs_view_1_softmaxed = teacher_patch_centered_sharpened[0]
        teacher_patch_embs_view_2_softmaxed = teacher_patch_centered_sharpened[1]

        mim_loss = self.loss.mim_loss(
            student_patch_embs_view_1_tokenized, 
            student_patch_embs_view_2_tokenized, 
            teacher_patch_embs_view_1_softmaxed, 
            teacher_patch_embs_view_2_softmaxed, 
            masks
        )

        loss = dino_loss + mim_loss

        return loss, mim_loss, dino_loss, teacher_embs, student_cls_before_head_mean, student_embs_avg_for_log