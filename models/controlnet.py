# models/controlnet.py
import torch
import torch.nn as nn
from models.unet import ResnetBlock, AttnBlock, Normalize, nonlinearity

class ControlNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch = config.model.ch  # 与UNet一致的基础通道数
        ch_mult = tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        self.resolution = config.data.image_size

        # 输入层：处理3通道骨架图像
        self.conv_in = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)

        # 下采样路径（与UNet下采样层级一一对应）
        self.down = nn.ModuleList()
        curr_res = self.resolution
        in_ch_mult = (1,) + ch_mult
        for i_level in range(len(ch_mult)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=0,  # ControlNet不使用时间步嵌入
                    dropout=config.model.dropout
                ))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != len(ch_mult) - 1:
                down.downsample = nn.Conv2d(block_in, block_in, 3, 2, 1)
                curr_res //= 2
            self.down.append(down)

        # 中间层
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=config.model.dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=0,
            dropout=config.model.dropout
        )

        # 特征映射层：确保控制特征与UNet特征通道数一致
        self.control_maps = nn.ModuleList()
        for i_level in range(len(ch_mult)):
            self.control_maps.append(nn.Conv2d(
                ch * ch_mult[i_level],
                ch * ch_mult[i_level],
                kernel_size=1, stride=1, padding=0
            ))

    def forward(self, x):
        """x: 3通道骨架图像"""
        h = self.conv_in(x)
        control_features = [h]  # 存储各层级控制特征

        # 下采样提取特征
        for i_level in range(len(self.down)):
            for i_block in range(len(self.down[i_level].block)):
                h = self.down[i_level].block[i_block](h, temb=None)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                control_features.append(h)
            if i_level != len(self.down) - 1:
                h = self.down[i_level].downsample(h)
                control_features.append(h)

        # 中间层特征
        h = self.mid.block_1(h, temb=None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb=None)
        control_features.append(h)

        # 映射到UNet通道数
        return [self.control_maps[i](f) for i, f in enumerate(control_features)]