import torch
import torch.nn as nn
import yaml

from bigdatasetGAN.biggan_pytorch import BigGAN
from transformer.cond_transformer import Net2NetTransformer
import torch.nn.functional as F
from training_utils import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data.dataloader import default_collate

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
                 transformer_ckpt=None,
                 transformer_config=None,
                 temperature=1.0,
                 top_k=100,
                 cuda=False):
        super(VQDatasetGAN, self).__init__()

        self.transformer_ckpt = transformer_ckpt
        self.transformer_config = transformer_config
        self.transformer_model = None
        self.resolution = resolution
        self.temperature = temperature
        self.top_k = top_k

        # Load Transformer including VQGAN
        with open(self.transformer_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)
            self.transformer_model = Net2NetTransformer(**config).eval()
            state_dict = torch.load(self.transformer_ckpt)["state_dict"]
            self.transformer_model.load_state_dict(state_dict, strict=True)

        self.low_feature_size = 32
        self.mid_feature_size = 128
        self.high_feature_size = 512

        low_feature_channel = 128
        mid_feature_channel = 64
        high_feature_channel = 32

    def _get_transformer_features(self, z_inds, c_inds):
        transformer_features = []
        cz = torch.cat([c_inds, z_inds], dim=1)
        token_embeddings = self.transformer_model.transformer.tok_emb(cz)
        t = token_embeddings.shape[1]
        position_embeddings = self.transformer_model.pos_emb[:, :t, :]
        x = self.transformer_model.transformer.drop(token_embeddings + position_embeddings)
        for idx, block in enumerate(self.transformer_model.transformer.blocks):
            if idx % 2 == 0:
                x = block(x)
                logits = self.transformer_model.transformer.ln_f(x)
                logits = self.transformer_model.transformer.head(x)
                logits = logits / self.temperature
                logits = self.transformer_model.top_k_logits(logits, self.top_k)
                probs = F.softmax(logits, dim=-1)
                _, ix = torch.topk(probs, 1, dim=-1)
                # Get rid of conditioning
                feature_ind = ix[:, z_inds.shape[1]-1:]
                feature_ind = self.transformer_model.permuter(feature_ind, reverse=True)
                bhwc = (z_inds.shape[0], z_inds.shape[2], z_inds.shape[3], z_inds.shape[1])
                features = self.transformer_model.first_stage_model.quantize.get_codebook_entry(feature_ind.reshape(-1), shape=bhwc)
                transformer_features.append(features)
        return features

    def forward(self, x, c):
        _, z_indices = self.transformer_model.encode_to_z(x)
        _, c_indices = self.transformer_model.encode_to_c(z_indices)
        features = self._get_transformer_features(z_indices, c_indices)



if __name__ == '__main__':
    params = {
        'resolution': 256,
        'out_dim': 256,
        'transformer_ckpt': '/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/research/surgical-tool-gen/Checkpoints/transformer/transformer_surgical_tools.ckpt',
        'transformer_config': 'bigdatasetGAN/config/transformer.yaml',
    }
    dataset_config_path = 'bigdatasetGAN/config/dataset.yaml'
    dataset_config = OmegaConf.load(dataset_config_path)
    dataset = instantiate_from_config(dataset_config.data)
    model =VQDatasetGAN(**params)
    # TODO: Get image from dataset for testing.