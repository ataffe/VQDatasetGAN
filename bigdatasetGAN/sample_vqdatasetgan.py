import argparse
import numpy as np
from torch.utils.data import DataLoader

from datasets.surgical_tool_dataset import SurgicalToolDataset
from vqdatasetgan import VQDatasetGAN
from pathlib import Path
import torch
from transformer.cond_transformer import Net2NetTransformer
from tqdm import trange
import cv2
import copy

def save_img(path: str, img: torch.Tensor, msk: torch.Tensor):
    img = img.detach().cpu()
    msk = msk.detach().cpu()

    # Unnormalize image
    img = ((img + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    msk = torch.where(msk > 0, 1, 0).permute(1,2,0).to(torch.uint8).numpy() * 255
    msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    msk = cv2.resize(msk, (512, 512), interpolation=cv2.INTER_NEAREST_EXACT)

    cv2.imwrite(path.replace(".jpg", "_original.jpg"), img)

    color_msk = copy.deepcopy(img)
    color_msk = np.where(msk == (255,255,255), (0,255,0), color_msk).astype(np.uint8)
    img = cv2.addWeighted(img, 0.5, color_msk, 0.5, 0)
    cv2.imwrite(path, img)

def get_quantized_input(transformer: Net2NetTransformer, loader: torch.utils.data.DataLoader, device: str):
    batch = next(iter(loader))
    batch["image"] = batch["image"].to(device)
    batch["coord"] = batch["coord"].to(device)
    x = transformer.get_input("image", batch).to(device)
    cond_key = transformer.cond_stage_key
    c = transformer.get_input(cond_key, batch).to(device)

    quant_z, z_indices = transformer.encode_to_z(x)
    quant_c, c_indices = transformer.encode_to_c(c)

    return quant_z, z_indices, quant_c, c_indices, batch["coord"]

def sample_transformer(
        model: VQDatasetGAN,
        z_indices: torch.Tensor,
        c_indices: torch.Tensor,
        z_latent_shape: tuple,
        c_latent_shape,
        sample: bool = False,
        topk: int = 5,
        temp: float = 1.0):
    idx = z_indices
    start = 0
    idx[:, start:] = 0
    idx = idx.reshape(z_latent_shape[0], z_latent_shape[2], z_latent_shape[3])
    start_i = start // z_latent_shape[3]
    start_j = start % z_latent_shape[3]
    cidx = c_indices
    cidx = cidx.reshape(c_latent_shape[0], c_latent_shape[2], c_latent_shape[3])
    for i in range(start_i, z_latent_shape[2] - 0):
        if i <= 8:
            local_i = i
        elif z_latent_shape[2] - i < 8:
            local_i = 16 - (z_latent_shape[2] - i)
        else:
            local_i = 8
        for j in range(start_j, z_latent_shape[3] - 0):
            if j <= 8:
                local_j = j
            elif z_latent_shape[3] - j < 8:
                local_j = 16 - (z_latent_shape[3] - j)
            else:
                local_j = 8

            i_start = i - local_i
            i_end = i_start + 16
            j_start = j - local_j
            j_end = j_start + 16
            patch = idx[:, i_start:i_end, j_start:j_end]
            patch = patch.reshape(patch.shape[0], -1)
            cpatch = cidx[:, i_start:i_end, j_start:j_end]
            cpatch = cpatch.reshape(cpatch.shape[0], -1)
            patch = torch.cat((cpatch, patch), dim=1)
            logits, _ = model.transformer_model.transformer(patch[:, :-1])
            logits = logits[:, -256:, :]
            logits = logits.reshape(z_latent_shape[0], 16, 16, -1)
            logits = logits[:, local_i, local_j, :]
            logits = logits / temp

            if topk is not None:
                logits = model.transformer_model.top_k_logits(logits, topk)

            # apply softmax to convert to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            idx[:, i, j] = ix[:, 0]
    return idx

def create_dataset(model: VQDatasetGAN, dataloader: torch.utils.data.DataLoader, in_args: argparse.Namespace):
    device = in_args.device
    model.eval()
    model.to(device)

    quant_z, z_indices, quant_c, c_indices, coord = get_quantized_input(model.transformer_model, dataloader, device)
    batch_size = in_args.batch_size
    num_images = in_args.num_images
    outdir = in_args.outdir
    image_count = 0

    # Generate Images
    for _ in trange(0, num_images, batch_size, desc="Generating Synthetic Dataset"):
        # Sample latent space entries from transformer
        sampled_indices = sample_transformer(
            model,
            z_indices,
            c_indices,
            quant_z.shape,
            quant_c.shape,
            in_args.sample,
            in_args.topk,
            in_args.temperature)
        # Decode quantized latent space to an image
        decoded_image = model.transformer_model.decode_to_img(
            sampled_indices[:, :quant_z.shape[2],:quant_z.shape[3]],
            quant_z.shape)

        # b,c,h,w -> b,h,w,c
        decoded_image = decoded_image.permute(0, 2, 3, 1)
        # Get Mask for sampled image
        mask = model({"image": decoded_image, "coord": coord})
        # Save image
        for idx in range(batch_size):
            save_img(outdir + f"/img{image_count}.jpg", decoded_image[idx, :, :, :], mask[idx, :, :, :])
            image_count += 1

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, help="Number of images to generate", default=1)
    parser.add_argument("--transformer_config", type=str, help="Path to transformer config.yaml")
    parser.add_argument("--transformer_ckpt", type=str, help="Path to transformer checkpoint.")
    parser.add_argument("--seg_model_ckpt", type=str, help="Path to model checkpoint.")
    parser.add_argument("--dataset_root", type=str, help="Path to dataset root.")
    parser.add_argument("--temperature", type=float, help="Sampling Temperature", default=1.0)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=8)
    parser.add_argument("--resolution", type=int, help="Resolution", default=256)
    parser.add_argument("--device", type=str, help="Device", default="cpu")
    parser.add_argument("--outdir", type=str, help="Output directory")
    parser.add_argument("--sample", type=bool, help="Sample transformer predictions from multinomial distribution", default=True)
    parser.add_argument("--topk", type=int, help="Top k predictions", default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Load dataset
    dataset = SurgicalToolDataset(root_dir=args.dataset_root, size=256)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Load Model
    model = VQDatasetGAN(
        resolution=args.resolution,
        out_dim=1,
        transformer_ckpt=args.transformer_ckpt,
        transformer_config=args.transformer_config)
    model.load_state_dict(torch.load(args.seg_model_ckpt, weights_only=True))

    # Create output directory if it does not exist
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    create_dataset(model, dataloader, args)





