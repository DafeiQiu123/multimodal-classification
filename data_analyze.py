import os
import json
import random
from collections import Counter

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

DATA_ROOT = r"C:\Users\Administrator\Desktop\toxic\data\hateful_meme"
IMG_DIR = os.path.join(DATA_ROOT, "img")

TOKENIZER_NAME = "distilbert-base-uncased"
MAX_LEN = 64


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def label_stats(items, split_name):
    labels = [x["label"] for x in items if "label" in x]
    if not labels:
        print(f"[{split_name}] no label field found.")
        return

    c = Counter(labels)
    total = len(labels)
    pos = c.get(1, 0)
    neg = c.get(0, 0)
    print(f"\n[{split_name}] label distribution:")
    print(f"  total={total}  neg(0)={neg}  pos(1)={pos}  pos_ratio={pos/total:.4f}")


def text_stats(items, split_name, tokenizer):
    texts = [x.get("text", "") for x in items]
    empty = sum(1 for t in texts if t is None or len(t.strip()) == 0)
    char_lens = np.array([len(t) if t else 0 for t in texts], dtype=np.int32)

    # token length (no truncation to see raw length distribution)
    tok_lens = []
    for t in texts[: min(len(texts), 5000)]:  # cap for speed
        if not t:
            tok_lens.append(0)
        else:
            tok_lens.append(len(tokenizer(t, add_special_tokens=True)["input_ids"]))
    tok_lens = np.array(tok_lens, dtype=np.int32)

    def describe(arr):
        return {
            "min": int(arr.min()) if len(arr) else None,
            "p25": int(np.percentile(arr, 25)) if len(arr) else None,
            "median": int(np.percentile(arr, 50)) if len(arr) else None,
            "p75": int(np.percentile(arr, 75)) if len(arr) else None,
            "p95": int(np.percentile(arr, 95)) if len(arr) else None,
            "max": int(arr.max()) if len(arr) else None,
            "mean": float(arr.mean()) if len(arr) else None,
        }

    print(f"\n[{split_name}] text stats:")
    print(f"  empty_text={empty}/{len(texts)}  empty_ratio={empty/max(len(texts),1):.4f}")
    print(f"  char_len  : {describe(char_lens)}")
    print(f"  token_len (first {len(tok_lens)} samples): {describe(tok_lens)}")

    # show a few extremes (by char length)
    idx_sorted = np.argsort(char_lens)
    for tag, idx in [("shortest", idx_sorted[0]), ("longest", idx_sorted[-1])]:
        ex = items[int(idx)]
        print(f"\n  [{split_name}] {tag} example:")
        print(f"    id={ex.get('id')} label={ex.get('label', None)}")
        print(f"    text={repr(ex.get('text',''))[:300]}")


def check_image_paths(items, split_name, k=30):
    # randomly verify image files exist
    picks = random.sample(items, min(k, len(items)))
    missing = 0
    for ex in picks:
        img_rel = ex.get("img", "")
        img_path = os.path.join(DATA_ROOT, img_rel)
        if not os.path.exists(img_path):
            missing += 1
    print(f"\n[{split_name}] image path check: checked={len(picks)} missing={missing}")


def visualize_samples(items, split_name, n=6, seed=0):
    random.seed(seed)
    picks = random.sample(items, min(n, len(items)))

    for i, ex in enumerate(picks):
        img_path = os.path.join(DATA_ROOT, ex["img"])
        text = ex.get("text", "")
        label = ex.get("label", None)
        _id = ex.get("id", None)

        print("\n" + "=" * 80)
        print(f"[{split_name}] sample {i+1}/{len(picks)}  id={_id}  label={label}")
        print(f"text: {text}")

        img = Image.open(img_path).convert("RGB")
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{split_name} id={_id} label={label}")
        plt.show()


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    train = load_jsonl(os.path.join(DATA_ROOT, "train.jsonl"))
    dev = load_jsonl(os.path.join(DATA_ROOT, "dev.jsonl"))
    test = load_jsonl(os.path.join(DATA_ROOT, "test.jsonl"))

    print("=== Split sizes ===")
    print(f"train: {len(train)}")
    print(f"dev  : {len(dev)}")
    print(f"test : {len(test)}")

    # label stats
    label_stats(train, "train")
    label_stats(dev, "dev")
    label_stats(test, "test")  # may have no labels

    # text stats
    text_stats(train, "train", tokenizer)
    text_stats(dev, "dev", tokenizer)
    text_stats(test, "test", tokenizer)

    # image existence check
    check_image_paths(train, "train")
    check_image_paths(dev, "dev")
    check_image_paths(test, "test")

    # visualize a few
    visualize_samples(train, "train", n=4, seed=42)
    visualize_samples(dev, "dev", n=4, seed=123)

    print("\nDone.")


if __name__ == "__main__":
    main()