import torch
import torch.nn as nn
import yaml
from vqgan.model.decoder import Decoder
from taming.modules.transformer.mingpt import GPT

class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, con_channels,
                 which_conv=nn.Conv2d, which_linear=None, activation=None,
                 upsample=None):
        super(SegBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_linear = which_conv, which_linear
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = BigGAN.layers.ccbn(in_channels, con_channels, self.which_linear, eps=1e-4, norm_style='bn')
        self.bn2 = BigGAN.layers.ccbn(out_channels, con_channels, self.which_linear, eps=1e-4, norm_style='bn')
        # upsample layers
        self.upsample = upsample

    def forward(self, x, y):
        h = self.activation(self.bn1(x, y))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h, y))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x

class VQDatasetGAN(nn.Module):
    def __init__(self,
                 resolution,
                 out_dim,
                 vqgan_ckpt=None,
                 transformer_ckpt=None,
                 vqgan_decoder_config=None,
                 transformer_config=None):
        super(VQDatasetGAN, self).__init__()

        self.vqgan_ckpt = vqgan_ckpt
        self.transformer_ckpt = transformer_ckpt
        self.vqgan_decoder_config = vqgan_decoder_config
        self.transformer_config = transformer_config
        self.vqgan_decoder_model = None
        self.transformer_model = None
        self.resolution = resolution
        # Load VQGAN Decoder and Transformer
        self._prepare_vqgan_decoder_model()
        self._prepare_transformer_model()

        self.low_feature_size = 32
        self.mid_feature_size = 128
        self.high_feature_size = 512



    def _prepare_vqgan_decoder_model(self):
        with open(self.vqgan_decoder_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            self.vqgan_decoder_model = Decoder(**config)
            state_dict = torch.load(self.vqgan_ckpt)
            self.vqgan_decoder_model.load_state_dict(state_dict, strict=False)


    def _prepare_transformer_model(self):
        with open(self.transformer_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            self.transformer_model = GPT(**config)
            state_dict = torch.load(self.transformer_ckpt)
            self.transformer_model.load_state_dict(state_dict, strict=False)


if __name__ == '__main__':
    params = {
        'resolution': 512,
        'out_dim': 512,
        'vqgan_ckpt': '/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/research/surgical-tool-gen/Checkpoints/vqgan/vqgan_surgical_tools.ckpt',
        'transformer_ckpt': '/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/research/surgical-tool-gen/Checkpoints/transformer/transformer_surgical_tools.ckpt',
        'vqgan_decoder_config': 'bigdatasetGAN/config/decoder.yaml',
        'transformer_config': 'bigdatasetGAN/config/transformer.yaml'
    }
    model =VQDatasetGAN(**params)