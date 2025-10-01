from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np
import json
import cv2
import torch

class SurgicalToolSegmentationDataset(Dataset):
    def __init__(self, data_path, img_size=256):
        self.image_size = img_size
        self.files = [str(path) for path in list(Path(data_path).rglob("*.png"))]
        self.annotations = {}
        self.label_names= {}
        try:
            annotation_file = list(Path(data_path).rglob("*.json"))[0]
        except Exception as e:
            raise FileNotFoundError("No annotations file found")

        with open(annotation_file, "r") as f:
            annotation_json = json.load(f)
            for category in annotation_json['categories']:
                self.label_names[category['id']] = category['name']
            images = {}
            for image in annotation_json['images']:
                images[image['id']] = image['file_name']

            for annotation in annotation_json['annotations']:
                annotations = self.annotations.get(images[annotation['image_id']], [])
                annotations.append(annotation)
                self.annotations[images[annotation['image_id']]] = annotations

    def __len__(self):
        return len(self.files)

    def preprocess_img(self, img_path):
        image = Image.open(img_path)
        image = image.convert("RGB")
        original_image_size = image.size
        if image.size != self.image_size:
            image = image.resize((self.image_size, self.image_size))
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        mask = self.get_mask(img_path, original_image_size)
        coord = np.arange(self.image_size * self.image_size).reshape(self.image_size, self.image_size, 1) / (self.image_size * self.image_size)
        return image, mask, coord

    def get_mask(self, img_path, img_size):
        try:
            annotations = self.annotations[Path(img_path).name]
        except Exception as e:
            raise RuntimeError(f"No annotations file found for {img_path}")
        mask = np.zeros(img_size, dtype=np.uint8)
        for annotation in annotations:
            polygon = np.array(annotation['segmentation'], dtype=np.int32).reshape(-1, 2)
            poly_mask = np.zeros(img_size, dtype=np.uint8)
            cv2.fillPoly(poly_mask, [polygon.astype(np.int32)], color=1)
            mask[poly_mask == 1] = 1
        mask = Image.fromarray(mask).convert("L")
        mask = mask.resize((self.image_size, self.image_size), resample=Image.Resampling.NEAREST)
        mask = np.array(mask).astype(np.uint8)
        return mask


    def __getitem__(self, idx):
        image, mask, coord = self.preprocess_img(self.files[idx])
        return {"image": torch.tensor(image, dtype=torch.float), "mask": torch.tensor(mask, dtype=torch.uint8), "coord": torch.tensor(coord, dtype=torch.float)}

def test_dataset():
    data_path = "/mnt/bddd2eea-89b7-45b0-8345-df09af140cd6/SSD/SurgicalToolDataset/SyntheticData/vqgan-v1/512x512/SegmentationDataset-v1"
    dataset = SurgicalToolSegmentationDataset(data_path)
    item = next(iter(dataset))
    assert item is not None
    assert item['image'].shape == (256, 256, 3), f"Image shape: {item['image'].shape} != (256, 256, 3)"
    assert item['mask'].shape == (256, 256), f"Mask shape: {item['mask'].shape} != (256, 256, 1)"
    mask = item['mask'].numpy()

    # Save Image
    image = ((item['image'] + 1.0) * 127.5).clip(0, 255).squeeze(0).to(torch.uint8)
    image = image.numpy()
    Image.fromarray(image).save("image.jpg")

    # Save Mask
    mask = np.where(mask > 0, 1, 0).astype(np.uint8)
    mask = mask * 255
    Image.fromarray(mask).save("mask.jpg")
    print("Passed Test!")

if __name__ == '__main__':
    test_dataset()
