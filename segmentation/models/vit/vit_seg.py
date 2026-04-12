import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.lora import LoRA
from segmentation.models.vit.layers import PatchEmbed
from segmentation.models.vit import VisionTransformer, init_vit
from segmentation.models.vit.decoders import (
    LinearDecoder,
    FPNDecoder,
    UperNetDecoder,
)

class ViTSegmentation(nn.Module):
    def __init__(
        self,
        r: int = 3,
        embed_dim: int = 384,
        num_classes: int = 4,
        use_lora: bool = False,
        decoder_cls: str = "linear",
        img_size: int = 1024,
        double_view: bool = False,
        in_chans: int = 2,
        freeze_encoder: bool = False,
        alpha: float = 16.0,
        n_layers: int = 4,
        **kwargs
    ):
        """
        Args:
            r (int, optional): The rank parameter of the LoRA weights. Defaults to 3.
            emb_dim (int, optional): The embedding dimension of the encoder. Defaults to 1024.
            num_classes (int, optional): The number of classes to output. Defaults to 1000.
            use_lora (bool, optional): Determines whether to use LoRA. Defaults to False.
            decoder_cls (str, optional): The decoder class for the segmentation model.
            img_size (int, optional): The input image dimension. Defaults to 512.
        """
        super().__init__()
        assert r > 0

        self.r = r
        self.alpha = alpha
        self.embed_dim = embed_dim
        self.img_dim = (img_size, img_size)
        self.use_lora = use_lora
        self.num_classes = num_classes
        self.double_view = double_view
        self.in_chans = in_chans
        self.freeze_encoder = freeze_encoder
        self.n_layers = n_layers

        # Select decoder class
        decoders = {
            "linear": LinearDecoder,
            "fpn": FPNDecoder,
            "upernet": UperNetDecoder,
        }
        self.decoder_cls = decoders[decoder_cls]

        # intialize model with encoder and decoder based on settings in yml file.
        
        # self.set_encoder(
        #     init_vit(
        #         size=kwargs["vit_type"],
        #         num_register_tokens=kwargs["reg_tokens"],
        #         **kwargs
        #     ),
        #     **kwargs
        # )
        self.set_decoder(**kwargs)


    def set_encoder(
        self,
        encoder: nn.Module,
        freeze_encoder: bool = False,
        use_dinov2=False,
        **kwargs
    ) -> None:
        # DINOv2 uses its own instantiation of the sample class
        if not use_dinov2:
            assert isinstance(
                encoder, VisionTransformer
            ), "The encoder is not of type `VisionTransformer`"

        # ============= Set the Encoder ==============
        self.encoder = encoder

        # Flexible patch encoder, and unfreeze
        if use_dinov2:
            self.encoder.patch_embed = PatchEmbed(
                img_size=(532, 532),
                patch_size=14,
                in_chans=self.in_chans,
                embed_dim=self.embed_dim,
            )
            for param in self.encoder.patch_embed.parameters():
                param.requires_grad = True

        # ===== Optionally add LoRA layers to the Encoder ==========
        if self.use_lora:
            print("Using LoRA with rank:", self.r)
            self.lora_layers = list(range(len(self.encoder.blocks)))
            self.w_a = []
            self.w_b = []

            # Freeze ALL encoder parameters first
            for param in self.encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze patch embedding for input adaptation
            if hasattr(self.encoder, 'patch_embed'):
                for param in self.encoder.patch_embed.parameters():
                    param.requires_grad = True
        
            # Unfreeze layer norm parameters (often helpful)
            for name, param in self.encoder.named_parameters():
                if 'norm' in name or 'ln' in name:
                    param.requires_grad = True

            for i, block in enumerate(self.encoder.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, self.r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, self.r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    alpha=self.alpha,
                )
            self._reset_lora_parameters()
        elif self.freeze_encoder:
            # If not using LoRA, we can still freeze the encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

            # Keep patch embedding trainable for input adaptation
            if use_dinov2 and hasattr(self.encoder, 'patch_embed'):
                for param in self.encoder.patch_embed.parameters():
                    param.requires_grad = True

    def set_decoder(self, **kwargs):
        # ============ Set the Decoder ==============
        self.decoder = self.decoder_cls(
            output_size=self.img_dim,  # output image dimensions
            num_classes=self.num_classes,  # Number of segmentation classes
            embed_dim=self.embed_dim,  # embedding_dimensions
            **kwargs  # decoder specific parameters
        )

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor]:
        assert self.encoder is not None, "First set an encoder before training"

        # If the FPN decoder is used, we take the n last layers for
        # our decoder to get a better segmentation result.
        if self.double_view:
            x_0 = x[:, 0, :, :].unsqueeze(1)
            x_1 = x[:, 1, :, :].unsqueeze(1)
            x = torch.cat((x_0, x_1), dim=0)
            assert self

        if isinstance(self.decoder, FPNDecoder):
            feature = self.encoder.get_intermediate_layers(
                x, n=self.inter_layers, reshape=True
            )
            # save embeddings to compute collapse
            embeds = feature[0]
            logits = self.decoder(feature)
        elif isinstance(self.decoder, UperNetDecoder):
            feature = self.encoder.get_intermediate_layers(x, n=self.n_layers, reshape=True)
            # save embeddings to compute collapse
            embeds = feature[0]
            # Decoder combines multiple layers of the encoder. Each layer's output is the patch tokens
            logits = self.decoder(feature) # [Layer1, Layer2, Layer3, Layer4]
            # Shape of logits is (batch_size, num_classes, height, width)

        else:  # We use the linear decoder
            feature_dict = self.encoder(x)

            # get the patch embeddings - so we exclude the CLS token
            # we also store them in variable to compute collapse
            patch_embeddings = embeds = feature_dict["x_norm_patchtokens"]
            
            # Shape of logits is (batch_size, num_classes, height, width)
            logits = self.decoder(patch_embeddings)
            logits = F.interpolate(
                logits,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits, embeds
