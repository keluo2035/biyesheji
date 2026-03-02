"""
模型定义：
  - UNet            : 标准 U-Net（实验 1、2）
  - AttentionMultiBranchUNet   : 注意力多分支融合（实验 3 核心）
  - MultiBranchNoAttentionUNet : 无注意力多分支（消融实验 4）
  - SharedEncoderAttentionUNet : 共享编码器+注意力（消融实验 5）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

# ======================== 基础模块 ========================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=True)
        return self.conv(torch.cat([x, skip], dim=1))


# ======================== 标准 U-Net ========================

class UNet(nn.Module):
    """用于实验 1（in_channels=1）和实验 2（in_channels=3）。"""

    def __init__(self, in_channels=1, num_classes=1, features=None):
        super().__init__()
        F_ = features or list(config.FEATURES)

        self.enc1 = EncoderBlock(in_channels, F_[0])
        self.enc2 = EncoderBlock(F_[0], F_[1])
        self.enc3 = EncoderBlock(F_[1], F_[2])
        self.enc4 = EncoderBlock(F_[2], F_[3])
        self.bottleneck = DoubleConv(F_[3], F_[3] * 2)

        self.dec4 = DecoderBlock(F_[3] * 2, F_[3], F_[3])
        self.dec3 = DecoderBlock(F_[3], F_[2], F_[2])
        self.dec2 = DecoderBlock(F_[2], F_[1], F_[1])
        self.dec1 = DecoderBlock(F_[1], F_[0], F_[0])
        self.out_conv = nn.Conv2d(F_[0], num_classes, 1)

    def forward(self, x):
        s1, d1 = self.enc1(x)
        s2, d2 = self.enc2(d1)
        s3, d3 = self.enc3(d2)
        s4, d4 = self.enc4(d3)
        b = self.bottleneck(d4)

        d = self.dec4(b, s4)
        d = self.dec3(d, s3)
        d = self.dec2(d, s2)
        d = self.dec1(d, s1)
        return self.out_conv(d)


# ======================== 编码器 / 解码器（供多分支模型复用） ========================

class UNetEncoder(nn.Module):
    """返回 [skip1, skip2, skip3, skip4, bottleneck]。"""

    def __init__(self, in_channels=1, features=None):
        super().__init__()
        F_ = features or list(config.FEATURES)
        self.enc1 = EncoderBlock(in_channels, F_[0])
        self.enc2 = EncoderBlock(F_[0], F_[1])
        self.enc3 = EncoderBlock(F_[1], F_[2])
        self.enc4 = EncoderBlock(F_[2], F_[3])
        self.bottleneck = DoubleConv(F_[3], F_[3] * 2)

    def forward(self, x):
        s1, d1 = self.enc1(x)
        s2, d2 = self.enc2(d1)
        s3, d3 = self.enc3(d2)
        s4, d4 = self.enc4(d3)
        b = self.bottleneck(d4)
        return [s1, s2, s3, s4, b]


class UNetDecoder(nn.Module):
    def __init__(self, num_classes=1, features=None):
        super().__init__()
        F_ = features or list(config.FEATURES)
        self.dec4 = DecoderBlock(F_[3] * 2, F_[3], F_[3])
        self.dec3 = DecoderBlock(F_[3], F_[2], F_[2])
        self.dec2 = DecoderBlock(F_[2], F_[1], F_[1])
        self.dec1 = DecoderBlock(F_[1], F_[0], F_[0])
        self.out_conv = nn.Conv2d(F_[0], num_classes, 1)

    def forward(self, skips, bottleneck):
        """skips: [s1, s2, s3, s4]"""
        d = self.dec4(bottleneck, skips[3])
        d = self.dec3(d, skips[2])
        d = self.dec2(d, skips[1])
        d = self.dec1(d, skips[0])
        return self.out_conv(d)


# ======================== 注意力融合模块 ========================

class ChannelAttention(nn.Module):
    """CBAM 风格通道注意力。"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.shared_fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_fc(F.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_fc(F.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 风格空间注意力。"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class AttentionFusionModule(nn.Module):
    """通道注意力 + 空间注意力 → 1×1 卷积降维。"""
    def __init__(self, in_channels, num_branches=3):
        super().__init__()
        total = in_channels * num_branches
        self.ca = ChannelAttention(total)
        self.sa = SpatialAttention()
        self.reduce = nn.Sequential(
            nn.Conv2d(total, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_list):
        x = torch.cat(feat_list, dim=1)
        x = x * self.ca(x)
        x = x * self.sa(x)
        return self.reduce(x)


class SimpleFusionModule(nn.Module):
    """拼接 + 1×1 卷积降维（无注意力，用于消融）。"""
    def __init__(self, in_channels, num_branches=3):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * num_branches, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feat_list):
        return self.reduce(torch.cat(feat_list, dim=1))


# ======================== 多分支融合 U-Net（通用基类） ========================

class MultiBranchFusionUNet(nn.Module):
    """
    多分支融合 U-Net。
      - use_attention=True  → 注意力融合
      - use_attention=False → 简单拼接融合
      - shared_encoder=True → 多分支共享同一编码器权重
    """

    def __init__(self, num_branches=3, num_classes=1,
                 features=None, use_attention=True,
                 shared_encoder=False):
        super().__init__()
        F_ = features or list(config.FEATURES)
        self.num_branches = num_branches
        self.shared_encoder = shared_encoder

        if shared_encoder:
            self.encoder = UNetEncoder(1, F_)
        else:
            self.encoders = nn.ModuleList(
                [UNetEncoder(1, F_) for _ in range(num_branches)])

        Fuse = AttentionFusionModule if use_attention else SimpleFusionModule
        self.skip_fusions = nn.ModuleList(
            [Fuse(f, num_branches) for f in F_])
        self.bn_fusion = Fuse(F_[-1] * 2, num_branches)

        self.decoder = UNetDecoder(num_classes, F_)

    def forward(self, inputs):
        """inputs: list[Tensor(B,1,H,W)]，长度 = num_branches"""
        if self.shared_encoder:
            all_feats = [self.encoder(x) for x in inputs]
        else:
            all_feats = [enc(x) for enc, x in zip(self.encoders, inputs)]

        fused_skips = []
        for lvl in range(4):
            lvl_feats = [all_feats[b][lvl] for b in range(self.num_branches)]
            fused_skips.append(self.skip_fusions[lvl](lvl_feats))

        bn_feats = [all_feats[b][4] for b in range(self.num_branches)]
        fused_bn = self.bn_fusion(bn_feats)
        return self.decoder(fused_skips, fused_bn)


# ======================== 子类（方便序列化 / 日志） ========================

class AttentionMultiBranchUNet(MultiBranchFusionUNet):
    """实验 3：独立编码器 + 注意力融合。"""
    def __init__(self, num_branches=3, **kw):
        super().__init__(num_branches, use_attention=True,
                         shared_encoder=False, **kw)


class MultiBranchNoAttentionUNet(MultiBranchFusionUNet):
    """实验 4 消融：独立编码器 + 简单拼接融合。"""
    def __init__(self, num_branches=3, **kw):
        super().__init__(num_branches, use_attention=False,
                         shared_encoder=False, **kw)


class SharedEncoderAttentionUNet(MultiBranchFusionUNet):
    """实验 5 消融：共享编码器 + 注意力融合。"""
    def __init__(self, num_branches=3, **kw):
        super().__init__(num_branches, use_attention=True,
                         shared_encoder=True, **kw)


# ======================== 工厂函数 ========================

_MODEL_MAP = {
    "unet": UNet,
    "attention_multibranch": AttentionMultiBranchUNet,
    "multibranch_no_attention": MultiBranchNoAttentionUNet,
    "shared_encoder_attention": SharedEncoderAttentionUNet,
}


def create_model(exp_id):
    """根据实验编号创建对应模型。"""
    cfg = config.EXPERIMENTS[exp_id]
    model_cls = _MODEL_MAP[cfg["model"]]
    if cfg["model"] == "unet":
        return model_cls(in_channels=len(cfg["sequences"]))
    else:
        return model_cls(num_branches=len(cfg["sequences"]))


def prepare_input(batch, exp_id, device):
    """根据实验类型，将 batch 转成模型所需的输入格式。"""
    cfg = config.EXPERIMENTS[exp_id]
    seqs = cfg["sequences"]
    if cfg["model"] == "unet":
        tensors = [batch[s].unsqueeze(1) for s in seqs]
        return torch.cat(tensors, dim=1).to(device)
    else:
        return [batch[s].unsqueeze(1).to(device) for s in seqs]
