import torch
import numpy as np
import torch.nn as nn

from segmentation.models.faster_vit.faster_vit_any_res import faster_vit_0_ssl_any_res
from segmentation.models.vit.layers import NestedTensorBlock as Block



class MaskedAutoencoderFasterVit(nn.Module):
    def __init__(self, **parameters):
        super().__init__()

        self.num_channels = int(parameters.get("in_chans", 2))
        self.img_size = int(parameters.get("img_size", 512))

        self.encoder = faster_vit_0_ssl_any_res(input_resolution=self.img_size)
        num_patches = self.encoder.num_patches

        # For the MAE decoder a fixed number of patches should be
        # followed.
        self.use_mae = parameters.get("use_mae", True)
        print("Using MAE decoder: ", self.use_mae)
        if self.use_mae:
            self.decoder = MaskedDecoder(
                num_patches=num_patches,
                embed_dim=self.encoder.embed_dim,
                patch_size=32, # Effective fixed patch size for the last level of FasterViT
                in_chans=parameters["in_chans"],
                depth=2,  # Taken of MAE paper
            )

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def patchify(self, imgs, patch_size=32):
        """
        Transforms images into patches.

        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        # [N, 3, H, W] -> [N, 3, H/p, p, W/p, p]
        x = imgs.reshape(imgs.shape[0], self.num_channels, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [N, h, w, p, p, C]
        x = x.reshape(imgs.shape[0], h * w, p**2 * self.num_channels)
        return x

    def unpatchify(self, x, patch_size=32):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], self.num_channels, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # [N, h, w, p, p, C]
        x = x.reshape(x.shape[0], h * w, p**2 * self.num_channels)
        return x

    def forward(self, x, mask_ratio, compute_loss=False):
        print("MAE x shape: ", x.shape)
        # Get patches for encoder
        x_patches = self.patchify(x)
        print("MAE x_patches shape: ", x_patches.shape)
        x_patches, masks, ids_restore = self.random_masking(
            x_patches, mask_ratio=mask_ratio
        )
        masks = masks.bool()

        x_ = self.encoder(x, masks)
        y_hat = self.decoder(x_["x_norm_patchtokens"], ids_restore)

        loss = None
        if compute_loss:
            y = self.patchify(x)

            loss = (y_hat - y) ** 2
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

            loss = (
                loss * masks
            ).sum() / masks.sum()  # mean loss on removed patches
        return loss, y_hat, masks, x_patches


class MaskedDecoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        patch_size: int = 32,
        in_chans: int = 2,
        depth: int = 2,
        decoder_embed: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        # Took similar settings to ViTDet

        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed)) # Try to use the same mask token for encoder and decoder
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed, bias=True)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(decoder_embed)
        # decoder to patch
        self.pred = nn.Linear(decoder_embed, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # predictor projection
        x = self.pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed
