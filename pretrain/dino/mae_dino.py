import torch

from segmentation.models.vit import init_vit
from segmentation.models.vit.downsample_vit import DownsampleVisionTransformer
from segmentation.models.ema import ModelEma
from pretrain.mae.mae import MaskedAutoencoder
from . import dino_loss
from .dino_head import DINOHead

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


class DinoMaskedAutoencoder(MaskedAutoencoder):

    def __init__(self, **parameters):
        super().__init__(**parameters)

        self.dino_loss_weight = parameters.get("dino_loss_weight", 1.0)
        self.reconstruction_loss_weight = parameters.get(
            "reconstruction_loss_weight", 0.0
        )
        self.ncrops = (
            parameters.get("nlocal_crops", 1) + 2
        )  # We add two because there are always 2 global crops
        self.normalize_loss = parameters.get("normalize_loss", False)
        self.centering_mode = parameters.get("centering_mode", None)
        self.donwsample_vit = parameters.get("downsample_vit", False)
        self.student_temp = parameters.get("student_temp", 0.1)
        self.teacher_temp = parameters.get("teacher_temp", 0.04)
        self.center_momentum = parameters.get("center_momentum", 0.9)
        self.embs_before_head = parameters.get("embs_before_head", False)
        self.use_projector = parameters.get("use_projector", True)

        if self.donwsample_vit:
            self.encoder = init_vit_tiny_downsample(
                parameters["vit_type"],
                num_register_tokens=self.num_reg_tokens,
                **parameters,
            )

        else:
            self.encoder = init_vit(
                parameters["vit_type"],
                num_register_tokens=self.num_reg_tokens,
                **parameters,
            )

        self.teacher_ema = ModelEma(
            self.encoder,
            decay=0.998,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.encoder_teacher = self.teacher_ema.module

        self.dino_loss = dino_loss.DINOLoss(
            student_temp=parameters["student_temp"],
            teacher_temp=self.teacher_temp,
            out_dim=parameters["out_dim"],
            center_momentum=self.center_momentum,
        )

        self.student_dino_head = (
            DINOHead(
                in_dim=self.encoder.embed_dim,
                out_dim=parameters["out_dim"],
            )
            if self.use_projector
            else None
        )

        self.teacher_dino_head = (
            DINOHead(
                in_dim=self.encoder.embed_dim,
                out_dim=parameters["out_dim"],
            )
            if self.use_projector
            else None
        )

        if self.normalize_loss:
            self.register_buffer("dino_loss_running_max", torch.tensor(float("-inf")))
            self.register_buffer("dino_loss_running_min", torch.tensor(float("inf")))
            self.register_buffer("mae_loss_running_max", torch.tensor(float("-inf")))
            self.register_buffer("mae_loss_running_min", torch.tensor(float("inf")))

        # self.use_mae declared in the parent class, otherwise
        # the decoder would be always initialized

    def forward_dino(self, x, encoder, head, embs_before_head=False):
        """Adapted code from Dino's MultiCropWrapper.
        Original Wrapper at the top of this file.
        """

        "Input : [B x crop1, B x crop2, ...]"
        if not isinstance(x, list):
            x = [x]

        # Groups crops of the same size together
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )  # Assumes that tensors of the same size are contiguous in the list

        start_idx = 0
        outputs = []
        for end_idx in idx_crops:
            inp = torch.cat(x[start_idx:end_idx])
            embs = encoder(inp)  # BxD
            outputs.append(embs["x_norm_clstoken"])
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        embs = None
        if embs_before_head:
            embs = outputs[0].detach().chunk(2)[0]
        outputs = torch.cat(outputs)
        if self.use_projector:
            outputs_after_head: torch.Tensor = head(outputs)
        else:
            outputs_after_head = outputs
        return outputs_after_head, embs

    def forward_mae(self, x, mask_ratio):

        x_patches = self.encoder.patch_embed(x)
        x_patches, masks, ids_restore = self.random_masking(
            x_patches, mask_ratio=mask_ratio
        )
        masks = masks.bool()

        x_ = self.encoder(x, masks)
        embs = x_["x_norm_patchtokens"]
        y_hat = self.decoder(embs, ids_restore)

        y = self.patchify(x)

        loss = (y_hat - y) ** 2
        loss = loss.mean()

        return loss, embs

    def update_running_min_max(
        self, loss: torch.Tensor, loss_min: torch.Tensor, loss_max: torch.Tensor
    ):
        with torch.no_grad():
            loss_max.copy_(torch.max(loss_max, loss.detach()))
            loss_min.copy_(torch.min(loss_min, loss.detach()))

    def forward(self, x, mask_ratio):
        """Forward pass of the model.

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

        loss = None
        embs_student = None
        embs_teacher = None

        x_globals = x[:2]

        student_cls_after_head, embs_student = self.forward_dino(
            x, self.encoder, self.student_dino_head, self.embs_before_head
        )
        student_cls_list = student_cls_after_head.chunk(self.ncrops)  # nlocal_crops + 2

        teacher_cls, embs_teacher = self.forward_dino(
            x_globals,
            self.encoder_teacher,
            self.teacher_dino_head,
            self.embs_before_head,
        )

        # embs = teacher_cls[0]

        # [2*B, D] -> [2*B, D]
        if self.centering_mode == "vanilla":
            teacher_cls_softmaxed = self.dino_loss.softmax_vanilla_teacher(teacher_cls)
            teacher_cls_list = teacher_cls_softmaxed.detach().chunk(2)
        elif self.centering_mode == "sharpen":
            teacher_cls_sharpened = self.dino_loss.softmax_sharpen_only_teacher(
                teacher_cls
            )
            teacher_cls_list = teacher_cls_sharpened.detach().chunk(2)
        elif self.centering_mode == "center":
            teacher_cls_centered = self.dino_loss.softmax_center_only_teacher(
                teacher_cls
            )
            teacher_cls_list = teacher_cls_centered.detach().chunk(2)
        else:
            teacher_cls_centered_sharpened = self.dino_loss.softmax_center_teacher(
                teacher_cls
            )
            teacher_cls_list = teacher_cls_centered_sharpened.detach().chunk(2)

        if not self.embs_before_head:
            embs_student = student_cls_list[0].detach()
            embs_teacher = teacher_cls_list[0]

        dino_loss = self.dino_loss(student_cls_list, teacher_cls_list)

        if self.normalize_loss:
            self.update_running_min_max(
                dino_loss, self.dino_loss_running_min, self.dino_loss_running_max
            )
            if self.dino_loss_running_max <= self.dino_loss_running_min:
                dino_loss = torch.tensor(0.0)
            else:
                dino_loss = (dino_loss - self.dino_loss_running_min) / (
                    self.dino_loss_running_max - self.dino_loss_running_min
                )

        loss = dino_loss * self.dino_loss_weight

        if self.use_mae:
            num_globals = len(x_globals)
            mae_loss = 0
            for x_glob in x_globals:
                curr_loss, _ = self.forward_mae(x_glob, mask_ratio)
                mae_loss += curr_loss
            mae_loss /= num_globals

            if self.normalize_loss:
                self.update_running_min_max(
                    mae_loss, self.mae_loss_running_min, self.mae_loss_running_max
                )
                if self.mae_loss_running_max <= self.mae_loss_running_min:
                    mae_loss = torch.tensor(0.0)
                else:
                    mae_loss = (mae_loss - self.mae_loss_running_min) / (
                        self.mae_loss_running_max - self.mae_loss_running_min
                    )

            loss += mae_loss * self.reconstruction_loss_weight

        return loss, embs_student, embs_teacher
