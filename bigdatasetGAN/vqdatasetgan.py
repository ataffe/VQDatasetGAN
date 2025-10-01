import torch
import torch.nn as nn
import yaml

from datasets.segmentation_dataset import SurgicalToolSegmentationDataset
from transformer.cond_transformer import Net2NetTransformer
import torch.nn.functional as F
from torch.utils.data import DataLoader
import functools

# Slimmed down version of batch norm from BigGan. Normal, non-class-conditional BN
class bn(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1):
        super(bn, self).__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.gain = nn.Parameter(torch.ones(output_size), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=True)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum
        # mean and variance
        self.register_buffer("stored_mean", torch.zeros(output_size))
        self.register_buffer("stored_var", torch.ones(output_size))

    def forward(self, x):
        return F.batch_norm(x, self.stored_mean, self.stored_var, self.gain,
                            self.bias, self.training, self.momentum, self.eps)


class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
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
        # self.bn1 = BigGAN.layers.ccbn(in_channels, con_channels, self.which_linear, eps=1e-4, norm_style='bn')
        # self.bn2 = BigGAN.layers.ccbn(out_channels, con_channels, self.which_linear, eps=1e-4, norm_style='bn')
        # self.bn1 = bn(in_channels)
        # self.bn2 = bn(out_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
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
                 top_k=100):
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

        self.transformer_model.eval()
        for parameter in self.transformer_model.parameters():
            parameter.requires_grad_(False)

        self.low_feature_size = 32
        self.mid_feature_size = 128
        self.high_feature_size = 256

        low_feature_channel = 128
        mid_feature_channel = 64
        high_feature_channel = 32

        which_conv = functools.partial(nn.Conv2d, kernel_size=3, padding=1)

        self.low_feature_conv = nn.Sequential(nn.Conv2d(3584, low_feature_channel, kernel_size=1, bias=False))
        self.mid_feature_conv = nn.Sequential(nn.Conv2d(512, mid_feature_channel, kernel_size=1, bias=False))

        self.mid_feature_mix_conv = SegBlock(
            in_channels=low_feature_channel + mid_feature_channel,
            out_channels=low_feature_channel + mid_feature_channel,
            which_conv=which_conv,
            which_linear=nn.Linear,
            activation=nn.ReLU(inplace=True),
            upsample=False,
        )

        self.high_feature_conv = nn.Sequential(
            nn.Conv2d(256, high_feature_channel, kernel_size=1, bias=False),
        )

        self.high_feature_mix_conv = SegBlock(
            in_channels=low_feature_channel + mid_feature_channel + high_feature_channel,
            out_channels=low_feature_channel + mid_feature_channel + high_feature_channel,
            which_conv=which_conv,
            which_linear=nn.Linear,
            activation=nn.ReLU(inplace=True),
            upsample=False,
        )

        # self.out_layer = nn.Conv2d(low_feature_channel + mid_feature_channel + high_feature_channel, out_dim, kernel_size=3, padding=1)
        self.out_layer = nn.Sequential(
            bn(low_feature_channel + mid_feature_channel + high_feature_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_feature_channel + mid_feature_channel + high_feature_channel, out_dim, kernel_size=3, padding=1)
        )

    def _indices_to_feature(self, indices, z_shape):
        logits = self.transformer_model.transformer.ln_f(indices)
        logits = self.transformer_model.transformer.head(logits)
        logits = logits / self.temperature
        logits = self.transformer_model.top_k_logits(logits, self.top_k)
        probs = F.softmax(logits, dim=-1)
        _, ix = torch.topk(probs, 1, dim=-1)
        # Get rid of conditioning
        feature_ind = ix[:, -256:, :]
        feature_ind = self.transformer_model.permuter(feature_ind, reverse=True)
        bhwc = (z_shape[0], z_shape[2], z_shape[3], z_shape[1])
        return self.transformer_model.first_stage_model.quantize.get_codebook_entry(feature_ind.reshape(-1),
                                                                                        shape=bhwc)

    def _get_transformer_features(self, z_inds: torch.Tensor, c_inds: torch.Tensor, z_shape: list[int]) -> list[torch.Tensor]:
        transformer_features = []
        cz = torch.cat([c_inds, z_inds], dim=1)
        token_embeddings = self.transformer_model.transformer.tok_emb(cz)
        t = token_embeddings.shape[1]
        position_embeddings = self.transformer_model.transformer.pos_emb[:, :t, :]
        x = self.transformer_model.transformer.drop(token_embeddings + position_embeddings)
        for idx, block in enumerate(self.transformer_model.transformer.blocks):
            if idx % 2 == 0:
                x = block(x)
                features = self._indices_to_feature(x, z_shape)
                transformer_features.append(features)
            else:
                x = block(x)
        return transformer_features


    def _get_decoder_features(self, x_encoded):
        decoder_features = []
        x = self.transformer_model.first_stage_model.post_quant_conv(x_encoded)
        decoder = self.transformer_model.first_stage_model.decoder
        temb = None
        h = decoder.conv_in(x)
        h = decoder.mid.block_1(h, temb)
        h = decoder.mid.attn_1(h)
        h = decoder.mid.block_2(h, temb)
        for i_level in reversed(range(decoder.num_resolutions)):
            for res_block in range(decoder.num_res_blocks + 1):
                h = decoder.up[i_level].block[res_block](h, temb)
                if len(decoder.up[i_level].attn) > 0:
                    h = decoder.up[i_level].attn[res_block](h)
            if i_level != 0:
                h = decoder.up[i_level].upsample(h)
            decoder_features.append(h)
        return decoder_features

    def _prepare_features(self, transformer_features, decoder_features, upsample='bilinear'):
        low_features = []
        # transformer features are [1, 256, 16, 16)
        for feature in transformer_features:
            low_features.append(F.interpolate(feature, self.low_feature_size, mode=upsample, align_corners=False))
        # First decoder feature is [1, 512, 32, 32]
        low_features.append(decoder_features[0])
        low_features = torch.cat(low_features, dim=1)

        mid_features = [
            F.interpolate(decoder_features[1], self.mid_feature_size, mode=upsample, align_corners=False),
            F.interpolate(decoder_features[2], self.mid_feature_size, mode=upsample, align_corners=False)
        ]
        mid_features = torch.cat(mid_features, dim=1)

        high_features = [
            F.interpolate(decoder_features[3], self.high_feature_size, mode=upsample, align_corners=False),
            F.interpolate(decoder_features[4], self.high_feature_size, mode=upsample, align_corners=False)
        ]
        high_features = torch.cat(high_features, dim=1)
        feature_dict = {
            "low": low_features,
            "mid": mid_features,
            "high": high_features
        }
        return feature_dict

    def forward(self, input):
        x = self.transformer_model.get_input("image", input)
        c = self.transformer_model.get_input("coord", input)
        quant_z, z_indices = self.transformer_model.encode_to_z(x)
        _, c_indices = self.transformer_model.encode_to_c(c)
        z_shape = quant_z.shape

        # Get features
        transformer_features = self._get_transformer_features(z_indices, c_indices,z_shape)
        decoder_features = self._get_decoder_features(transformer_features[-1])
        feature_dict = self._prepare_features(transformer_features, decoder_features)

        # Low  Features
        low_feat = self.low_feature_conv(feature_dict["low"])
        low_feat = F.interpolate(low_feat, self.mid_feature_size, mode="bilinear", align_corners=False)

        # Mid Features
        mid_feat = self.mid_feature_conv(feature_dict["mid"])
        mid_feat = torch.cat([low_feat, mid_feat], dim=1)
        mid_feat = self.mid_feature_mix_conv(mid_feat)
        mid_feat = F.interpolate(mid_feat, size=self.high_feature_size, mode="bilinear", align_corners=False)

        # High Features
        high_feat = self.high_feature_conv(feature_dict["high"])
        high_feat = torch.cat([mid_feat, high_feat], dim=1)
        high_feat = self.high_feature_mix_conv(high_feat)
        return self.out_layer(high_feat)

def test_model():
    params = {
        'resolution': 256,
        'out_dim': 1,
        'transformer_ckpt': '/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/research/surgical-tool-gen/Checkpoints/transformer/transformer_surgical_tools.ckpt',
        'transformer_config': 'bigdatasetGAN/config/transformer.yaml',
    }
    dataset_path = "/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/SSD/SurgicalToolDataset/SyntheticData/vqgan-v1/512x512/SegmentationDataset-v1"
    device = 'cuda'
    train_dataset = SurgicalToolSegmentationDataset(dataset_path)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    item = next(iter(dataloader))
    item["image"] = item["image"].to(device)
    item["coord"] = item["coord"].to(device)
    item["mask"] = item["mask"].to(device)
    model = VQDatasetGAN(**params).to(device)
    model.eval()
    mask = model(item)
    assert mask.shape[0] == 1
    assert mask.shape[1] == 1
    assert mask.shape[2] == 256
    assert mask.shape[3] == 256
    print("Test passed!")



if __name__ == '__main__':
    test_model()
