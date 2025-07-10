from pathlib import Path
import argparse

def make_train_test_split(input_dir, train_split=80):
    input_dir = Path(input_dir)
    images = list(input_dir.glob("*.jpg"))

    if not images:
        print(f"No images found in {input_dir}.")
        return

    print(f'Found {len(images)} images in {input_dir}.')

    train_count = int(len(images) * train_split / 100)
    train_images = images[:train_count]
    test_images = images[train_count:]

    train_dir = input_dir / "train"
    test_dir = input_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for img in train_images:
        img.rename(train_dir / img.name)

    for img in test_images:
        img.rename(test_dir / img.name)

    print(f"Split {len(images)} images into {len(train_images)} training and {len(test_images)} testing images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Directory containing the images.")
    parser.add_argument("--train-split", type=int, default=80, help="Percentage of images to use for training (default: 80).")
    args = parser.parse_args()
    make_train_test_split(args.input_dir, args.train_split)