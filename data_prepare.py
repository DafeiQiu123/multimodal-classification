import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HatefulMemesDataset(Dataset):
    """
    Reads local train/dev/test jsonl and images.
    Each row: {"id":..., "img":"img/xxxxx.png", "text":"...", "label":0/1}
    Note: test.jsonl may not have label depending on source.
    """
    def __init__(self, data_root, split="train", tokenizer=None, max_len=64, image_size=224):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.tokenizer = tokenizer
        self.max_len = max_len

        jsonl_path = os.path.join(data_root, f"{split}.jsonl")
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

        self.img_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        img_path = os.path.join(self.data_root, ex["img"])  # "img/xxx.png"
        image = Image.open(img_path).convert("RGB")
        image = self.img_tf(image)

        text = ex.get("text", "")
        label = ex.get("label", -1)  # test may not have label

        return {
            "image": image,
            "text": text,
            "label": int(label),
            "id": ex.get("id", idx),
        }

# def make_collate_fn(tokenizer, max_len=64):
#     """
#     Dynamic padding: tokenize the batch texts together.
#     """
#     def collate(batch):
#         images = torch.stack([b["image"] for b in batch], dim=0)
#         labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
#         texts = [b["text"] for b in batch]
#         ids = [b["id"] for b in batch]

#         enc = tokenizer(
#             texts,
#             padding=True,
#             truncation=True,
#             max_length=max_len,
#             return_tensors="pt",
#         )

#         return {
#             "images": images,
#             "input_ids": enc["input_ids"],
#             "attention_mask": enc["attention_mask"],
#             "labels": labels,
#             "texts": texts,
#             "ids": ids,
#         }
#     return collate
    # data_prepare.py
class Collator:
    def __init__(self, tokenizer, max_len=64):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch], dim=0)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        texts  = [b["text"] for b in batch]
        ids    = [b["id"] for b in batch]

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "images": images,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "texts": texts,
            "ids": ids,
        }