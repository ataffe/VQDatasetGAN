import argparse
import os.path

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from bigdatasetGAN.vqdatasetgan import VQDatasetGAN
from datasets.segmentation_dataset import SurgicalToolSegmentationDataset
from datetime import datetime
from pathlib import Path
import torch
from torchvision.utils import make_grid
from torchinfo import summary

def make_training_dir(save_dir):
    current_time = datetime.now().strftime('%b%d_%H-%M')
    run_dir = os.path.join(args.save_dir, f'run-{current_time}')
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    log_dir = os.path.join(run_dir, 'logs')
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, log_dir

def train(args):
    device = "cuda" if args.use_cuda else "cpu"
    # Setup training folder
    run_dir, ckpt_dir, log_dir = make_training_dir(args.save_dir)
    log_writer = SummaryWriter(log_dir=log_dir)
    print(f"\nTraining run folder created: {run_dir}")
    print(f"====> Checkpoints: {ckpt_dir}")
    print(f"====> Logs: {log_dir}")

    # Create Training Dataset
    train_dataset = SurgicalToolSegmentationDataset(args.dataset_path, args.image_size)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Data loaded: {len(train_dataset)} images found")

    # Load Model
    print(f"\nLoading Model")
    model = VQDatasetGAN(
        resolution=args.image_size,
        out_dim=1,
        transformer_ckpt=args.transformer_ckpt,
        transformer_config=args.transformer_config).to(device)

    print("Model Summary")
    summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    print("Starting training with the following parameters:")
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.learning_rate)
    global_step = 0
    for epoch in range(args.epochs):
        min_loss = None
        loss = None
        last_batch = None
        last_mask = None
        print(f"#### Epoch: {epoch} ####")
        for iteration, batch in enumerate(dataloader):
            if iteration == args.epochs:
                break

            model.train()
            model.transformer_model.eval()

            batch["image"] = batch["image"].to(device)
            batch["coord"] = batch["coord"].to(device)
            batch["mask"] = batch["mask"].to(device)

            # Run model
            mask_pred = model(batch)

            last_mask = mask_pred
            gt_mask = batch["mask"].float().unsqueeze(1)
            loss = loss_fn(mask_pred, gt_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_writer.add_scalar("train/loss", loss.item(), global_step=global_step)
            if iteration % 10 == 0:
                print("Training step: {0:05d}/{1:05d}, loss: {2:0.4f}".format(iteration, args.epochs, loss))
            last_batch = batch
            global_step += 1

        if (min_loss is None or loss < min_loss) and epoch % 10 == 0:
            min_loss = loss
            print("Saving checkpoint")
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "checkpoint-latest.pth".format(epoch)))

        if epoch % 10 == 0:
            # Log image
            images = last_batch["image"].detach().cpu()
            unnormalized_img = ((images + 1.0) * 127.5).clip(0, 255).squeeze(0).permute(2, 0, 1).to(torch.uint8)
            image_grid = make_grid(unnormalized_img, nrow=1, padding=3)
            log_writer.add_image("train/images_gt", image_grid, global_step=epoch)

            # Log Ground Truth Mask
            mask = last_batch["mask"].detach().cpu()
            mask = torch.where(mask > 0, 1, 0)
            mask = mask * 255
            image_grid = make_grid(mask, nrow=1, padding=3)
            log_writer.add_image("train/mask_gt", image_grid, global_step=epoch)

            # Log Predicted Mask
            pred_mask = last_mask.detach().cpu()
            pred_mask = torch.where(pred_mask > 0, 1, 0)
            pred_mask = pred_mask * 255
            image_grid = make_grid(pred_mask, nrow=1, padding=3)
            log_writer.add_image("train/pred_mask", image_grid, global_step=epoch)
    print("Training complete.")


def parse_args():
    parser = argparse.ArgumentParser(description='Trains VQDatasetGAN model')
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--transformer_ckpt", type=str, required=True)
    parser.add_argument("--transformer_config", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)