import os
import json
import random
from PIL import Image
import matplotlib.pyplot as plt

# ====== 修改为你的本地数据路径 ======
DATA_ROOT = r"C:\Users\Administrator\Desktop\toxic\data\hateful_meme"
IMG_DIR = os.path.join(DATA_ROOT, "img")
TRAIN_JSON = os.path.join(DATA_ROOT, "train.jsonl")
DEV_JSON = os.path.join(DATA_ROOT, "dev.jsonl")
TEST_JSON = os.path.join(DATA_ROOT, "test.jsonl")


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def show_samples(json_path, num_samples=3):
    data = load_jsonl(json_path)
    print(f"\nLoaded {len(data)} samples from {json_path}")

    samples = random.sample(data, num_samples)

    for i, sample in enumerate(samples):
        img_rel_path = sample["img"]   # e.g. "img/42953.png"
        img_path = os.path.join(DATA_ROOT, img_rel_path)

        text = sample["text"]
        label = sample.get("label", None)

        print("=" * 60)
        print(f"Sample {i+1}")
        print(f"Image: {img_path}")
        print(f"Text : {text}")
        print(f"Label: {label}")

        # 显示图片
        img = Image.open(img_path).convert("RGB")
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title(f"Label={label}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    print("Exploring TRAIN split")
    show_samples(TRAIN_JSON, num_samples=5)

    print("\nExploring DEV split")
    show_samples(DEV_JSON, num_samples=5)