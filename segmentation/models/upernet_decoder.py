import torch
from torch import nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=has_bias,
    )


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class UperNetDecoder(nn.Module):
    def __init__(
        self,
        output_size: tuple[int] = (1024, 1024),  # resolution of output image
        num_classes=4,  # number of segmentation classes
        pool_mode="max",
        pool_scales=(2, 4, 8, 16),
        inter_dim=256,
        embed_dim=384,
        fpn_inplanes=None,
        **kwargs
    ):
        super().__init__()
        self.resolution = output_size
        self.num_classes = num_classes
        self.ppm_dim = embed_dim

        fc_dim = self.ppm_dim // len(pool_scales)
        self.ppm_out_dim = self.ppm_dim + len(pool_scales) * fc_dim

        # Set fpn_inplanes to encoder intermediate layer channel dimension.
        # For vit model this is always 512 (embed_dim).
        if fpn_inplanes is not None:
            self.fpn_inplanes = fpn_inplanes
        else:
            # Default to embed_dim for all pool scales.
            # This is the case for ViT models.
            self.fpn_inplanes = (embed_dim,) * len(pool_scales)

        # Using different pooling method, no PrRoIPool2D, just
        # average, max pooling. Need PrRoIPool2D for pool scale 1.
        self.pool = nn.AdaptiveMaxPool2d if pool_mode == "max" else nn.AdaptiveAvgPool2d

        # PPM Module
        self.ppm = []
        fc_dim = self.ppm_dim // len(pool_scales)  # Heleen -> 192
        self.ppm_out_dim = self.ppm_dim + len(pool_scales) * fc_dim
        for scale in pool_scales:
            self.ppm.append(
                nn.Sequential(
                    self.pool(scale),
                    nn.Conv2d(self.ppm_dim, fc_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fc_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.ppm = nn.ModuleList(self.ppm)

        self.ppm_last_conv = conv3x3_bn_relu(self.ppm_out_dim, inter_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in self.fpn_inplanes[:-1]:  # skip the top layer
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, inter_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(inter_dim),
                    nn.ReLU(inplace=True),
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(self.fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(
                nn.Sequential(
                    conv3x3_bn_relu(inter_dim, inter_dim, 1),
                )
            )
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(
            len(self.fpn_inplanes) * inter_dim, inter_dim, 1
        )

        # Optional from Fusion or FPN_2 out.
        # The input dim: is fpn_dim
        self.head = nn.Sequential(
            conv3x3_bn_relu(inter_dim, inter_dim, 1),
            nn.Conv2d(inter_dim, self.num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        # PPM module
        out = x[-1]

        H, W = out.shape[-2], out.shape[-1]
        ppm_out = [out]
        for conv_block in self.ppm:
            ppm_out.append(F.interpolate(conv_block(out), size=(H, W), mode="bilinear"))
        ppm_out = torch.cat(ppm_out, 1)
        ppm_out = self.ppm_last_conv(ppm_out)

        # FPN Module
        fpn_out = [ppm_out]
        for i in reversed(range(len(x) - 1)):
            conv_x = self.fpn_in[i](x[i])  # lateral branch

            ppm_out = F.interpolate(
                ppm_out,
                size=conv_x.size()[2:],
                mode="bilinear",
                align_corners=False,
            )  # top-down branch
            ppm_out = conv_x + ppm_out

            fpn_out.append(self.fpn_out[i](ppm_out))
        fpn_out.reverse()  # [P2 - P5]

        # Fusion of FPN
        output_size = fpn_out[0].size()[2:]
        fusion_list = [fpn_out[0]]
        for i in range(1, len(fpn_out)):
            fusion_list.append(
                F.interpolate(
                    fpn_out[i], output_size, mode="bilinear", align_corners=False
                )
            )
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_fusion(fusion_out)

        x = self.head(x)

        return F.interpolate(x, self.resolution, mode="bilinear", align_corners=False)
