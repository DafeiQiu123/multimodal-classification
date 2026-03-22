import os
import json
import argparse
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from model_v3 import CLIPFusionClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../data/hateful_meme")
    parser.add_argument("--clip_backbone", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--stage1_epochs", type=int, default=8)   # frozen encoder epochs
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--encoder_lr", type=float, default=5e-7)
    parser.add_argument("--fusion_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=77)
    parser.add_argument("--no_freeze_vision", action="store_true")
    parser.add_argument("--no_freeze_text", action="store_true")
    parser.add_argument("--unfreeze_vision_last_n_layers", type=int, default=1)
    parser.add_argument("--unfreeze_text_last_n_layers", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


# ---- Augmentation pipeline (PIL -> PIL, CLIPProcessor handles the rest) ----
TRAIN_AUG = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomRotation(degrees=10),
])


# ---- Dataset: returns raw PIL images (CLIPProcessor handles all preprocessing) ----
class CLIPMemesDataset(Dataset):
    def __init__(self, data_root, split, augment=False):
        self.data_root = data_root
        self.augment = augment
        jsonl_path = os.path.join(data_root, f"{split}.jsonl")
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        img_path = os.path.join(self.data_root, ex["img"])
        image = Image.open(img_path).convert("RGB")
        if self.augment:
            image = TRAIN_AUG(image)
        return {
            "image": image,
            "text": ex.get("text", ""),
            "label": int(ex.get("label", -1)),
            "id": ex.get("id", idx),
        }


# ---- Collator: uses CLIPProcessor for both image and text ----
class CLIPCollator:
    def __init__(self, processor, max_len=77):
        self.processor = processor
        self.max_len = max_len

    def __call__(self, batch):
        images = [b["image"] for b in batch]
        texts = [b["text"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        enc = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        return {
            "pixel_values": enc["pixel_values"],
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def compute_class_weights(dataset, device):
    labels = [x["label"] for x in dataset.items]
    n = len(labels)
    n0 = labels.count(0)
    n1 = labels.count(1)
    w0 = n / (2 * n0)
    w1 = n / (2 * n1)
    print(f"[class weights] n={n} n0={n0} n1={n1} -> w0={w0:.4f} w1={w1:.4f}")
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    preds, labels_all = [], []
    total_loss, total_correct, n = 0.0, 0, 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(pixel_values, input_ids, attention_mask)
        loss = loss_fn(logits, y)
        pred = logits.argmax(dim=1)

        preds.extend(pred.cpu().numpy())
        labels_all.extend(y.cpu().numpy())
        total_loss += loss.item() * y.size(0)
        total_correct += (pred == y).sum().item()
        n += y.size(0)

    acc = total_correct / n
    f1 = f1_score(labels_all, preds)
    precision = precision_score(labels_all, preds, zero_division=0)
    recall = recall_score(labels_all, preds, zero_division=0)
    cm = confusion_matrix(labels_all, preds)
    return total_loss / n, acc, f1, precision, recall, cm


def trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def set_encoder_grad(model, unfreeze_vision_n=0, unfreeze_text_n=0):
    """Freeze all encoder params, then selectively unfreeze last N layers."""
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.text_encoder.parameters():
        p.requires_grad = False

    if unfreeze_vision_n > 0:
        vis_layers = model.vision_encoder.vision_model.encoder.layers
        for layer in vis_layers[-unfreeze_vision_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(model.vision_encoder.vision_model, "post_layernorm"):
            for p in model.vision_encoder.vision_model.post_layernorm.parameters():
                p.requires_grad = True

    if unfreeze_text_n > 0:
        txt_layers = model.text_encoder.text_model.encoder.layers
        for layer in txt_layers[-unfreeze_text_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        if hasattr(model.text_encoder.text_model, "final_layer_norm"):
            for p in model.text_encoder.text_model.final_layer_norm.parameters():
                p.requires_grad = True


def build_optimizer_and_scheduler(model, args, n_steps):
    vis_params = trainable_params(model.vision_encoder)
    txt_params = trainable_params(model.text_encoder)
    fusion_params = (
        trainable_params(model.vis_proj)
        + trainable_params(model.cross_attn)
        + trainable_params(model.attn_ln)
        + trainable_params(model.ffn)
        + trainable_params(model.ffn_ln)
    )
    cls_params = trainable_params(model.classifier)

    param_groups = [
        {"params": fusion_params, "lr": args.fusion_lr},
        {"params": cls_params, "lr": args.classifier_lr},
    ]
    if vis_params:
        param_groups.insert(0, {"params": vis_params, "lr": args.encoder_lr})
    if txt_params:
        param_groups.insert(0, {"params": txt_params, "lr": args.encoder_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    warmup = int(n_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, n_steps)

    print(
        f"[trainable #params] "
        f"vision={sum(p.numel() for p in vis_params):,} "
        f"text={sum(p.numel() for p in txt_params):,} "
        f"fusion={sum(p.numel() for p in fusion_params):,} "
        f"classifier={sum(p.numel() for p in cls_params):,}"
    )
    return optimizer, scheduler


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("outputs", exist_ok=True)

    processor = CLIPProcessor.from_pretrained(args.clip_backbone)

    train_ds = CLIPMemesDataset(args.data_root, "train", augment=True)
    dev_ds = CLIPMemesDataset(args.data_root, "dev", augment=False)

    collator = CLIPCollator(processor, max_len=args.max_len)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers,
        generator=g, pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Always start with fully frozen encoders
    model = CLIPFusionClassifier(
        clip_backbone=args.clip_backbone,
        freeze_vision=True,
        freeze_text=True,
    ).to(args.device)

    class_weights = compute_class_weights(train_ds, args.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    stage2_epochs = args.epochs - args.stage1_epochs
    print(f"[two-stage] stage1={args.stage1_epochs} epochs (frozen encoders) | "
          f"stage2={stage2_epochs} epochs (unfreeze last {args.unfreeze_vision_last_n_layers}v / "
          f"{args.unfreeze_text_last_n_layers}t layers)")
    print(f"[lr groups] encoder_lr={args.encoder_lr} fusion_lr={args.fusion_lr} "
          f"classifier_lr={args.classifier_lr}")

    # ---- Stage 1: frozen encoders ----
    print("\n=== Stage 1: frozen encoders ===")
    set_encoder_grad(model, 0, 0)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args, n_steps=args.stage1_epochs * len(train_loader)
    )

    best_f1 = -1.0

    def run_epoch(epoch, total_epochs, optimizer, scheduler):
        nonlocal best_f1
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{total_epochs}")
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pred = logits.argmax(dim=1)
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (pred == labels).sum().item()
            total_n += bs
            pbar.set_postfix(loss=total_loss / max(total_n, 1), acc=total_correct / max(total_n, 1))

        dev_loss, dev_acc, dev_f1, prec, rec, cm = evaluate(model, dev_loader, args.device, loss_fn)
        print(
            f"[epoch {epoch}/{total_epochs}] "
            f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} dev_f1={dev_f1:.4f} "
            f"precision={prec:.4f} recall={rec:.4f}"
        )
        print("confusion_matrix [[TN, FP], [FN, TP]]")
        print(cm)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), "outputs/best_v3.pt")
            print(f"[saved] best model (f1={best_f1:.4f})")

    for epoch in range(1, args.stage1_epochs + 1):
        run_epoch(epoch, args.epochs, optimizer, scheduler)

    # ---- Stage 2: unfreeze last N encoder layers ----
    if stage2_epochs > 0:
        print("\n=== Stage 2: partial encoder unfreeze ===")
        set_encoder_grad(model, args.unfreeze_vision_last_n_layers, args.unfreeze_text_last_n_layers)
        optimizer, scheduler = build_optimizer_and_scheduler(
            model, args, n_steps=stage2_epochs * len(train_loader)
        )
        for epoch in range(args.stage1_epochs + 1, args.epochs + 1):
            run_epoch(epoch, args.epochs, optimizer, scheduler)

    print(f"\nTraining finished. Best dev F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
