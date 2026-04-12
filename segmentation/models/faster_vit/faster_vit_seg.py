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
from segmentation.models.faster_vit.faster_vit_any_res import FasterViT

class FasterVitSeg(nn.Module):
    def __init__(
        self,
        encoder: FasterViT,
        r: int = 3,
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

        # Feature shape: torch.Size([64, 128, 64, 64])
        # Feature shape: torch.Size([64, 256, 32, 32])
        # Feature shape: torch.Size([64, 512, 16, 16])
        # Feature shape: torch.Size([64, 512, 16, 16])

        self.encoder = self.set_encoder(
            encoder,
            freeze_encoder=freeze_encoder,
            **kwargs
        )
        encoder_dim = self.encoder.dim

        

        multi_list = [2 ** (i+1) for i in range(len(self.encoder.levels))]
        multi_list[-1] = multi_list[-2] # Last one is the same as the second last
        fpn_inplanes = [encoder_dim * m for m in multi_list]
        pool_scales = (2, 4, 8, 16)
        


        self.decoder = UperNetDecoder(
            embed_dim=self.encoder.embed_dim,
            output_size=(img_size, img_size),
            num_classes=num_classes,
            fpn_inplanes=fpn_inplanes,
            pool_scales=pool_scales,
        )



    def set_encoder(
        self,
        encoder: FasterViT,
        freeze_encoder: bool = False,
        **kwargs
    ) -> FasterViT:

        # ============= Set the Encoder ==============
        self.freeze_encoder = freeze_encoder

        # ===== Optionally add LoRA layers to the Encoder ==========
        if self.use_lora:
            encoder = self.setup_lora(encoder)
        elif self.freeze_encoder:
            # If not using LoRA, we can still freeze the encoder
            for param in encoder.parameters():
                param.requires_grad = False
        
        return encoder

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

        if isinstance(self.decoder, FPNDecoder):
            feature = self.encoder.forward_intermediate(x)
            logits = self.decoder(feature)
            embeds = None
        elif isinstance(self.decoder, UperNetDecoder):
            feature = self.encoder.forward_intermediate(x)
            for f in feature:
                print(f"Feature shape: {f.shape}")
            # Decoder combines multiple layers of the encoder. Each layer's output is the patch tokens
            logits = self.decoder(feature) # [Layer1, Layer2, Layer3, Layer4]
            # Shape of logits is (batch_size, num_classes, height, width)
            embeds = None

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

    def setup_lora(self, encoder:FasterViT):
            print("Using LoRA with rank:", self.r)
            self.lora_layers = list(range(len(encoder.blocks)))
            self.w_a = []
            self.w_b = []

            # Freeze ALL encoder parameters first
            for param in encoder.parameters():
                param.requires_grad = False
            
            # Unfreeze patch embedding for input adaptation
            if hasattr(encoder, 'patch_embed'):
                for param in encoder.patch_embed.parameters():
                    param.requires_grad = True
        
            # Unfreeze layer norm parameters (often helpful)
            for name, param in encoder.named_parameters():
                if 'norm' in name or 'ln' in name:
                    param.requires_grad = True

            for i, block in enumerate(encoder.blocks):
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
            return encoder