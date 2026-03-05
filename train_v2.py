# train_v2.py
import os
import argparse
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from data_prepare import HatefulMemesDataset, Collator
from model_v2 import FusionXAttnBinaryClassifier


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=r"C:\Users\Administrator\Desktop\toxic\data\hateful_meme")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)

    # NOTE: lr arg kept for compatibility, but optimizer uses grouped LRs below.
    p.add_argument("--lr", type=float, default=1e-5)

    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="./outputs")

    p.add_argument("--freeze_image", action="store_true")
    p.add_argument("--freeze_text", action="store_true")

    p.add_argument("--num_workers", type=int, default=2)

    # optional: make unfreeze knobs configurable
    p.add_argument("--unfreeze_resnet_last_stage", action="store_true")
    p.add_argument("--unfreeze_bert_last_n_layers", type=int, default=0)

    return p.parse_args()


def compute_class_weights_from_dataset(train_ds, device):
    """
    Compute class weights for CrossEntropyLoss:
        w_c = N / (K * n_c)
    """
    labels = [it.get("label", None) for it in train_ds.items]
    labels = [l for l in labels if l is not None and l in (0, 1)]

    n = len(labels)
    n0 = sum(1 for l in labels if l == 0)
    n1 = sum(1 for l in labels if l == 1)

    if n0 == 0 or n1 == 0:
        print("[warn] One class has 0 samples. Fallback to uniform weights.")
        return torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    w0 = n / (2.0 * n0)
    w1 = n / (2.0 * n1)

    w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    print(f"[class weights] n={n} n0={n0} n1={n1} -> w0={w0:.4f} w1={w1:.4f}")
    return w


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0

    all_preds = []
    all_labels = []

    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(images, input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        preds = logits.argmax(dim=1)

        bs = labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * bs
        n += bs

        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    acc = total_correct / max(n, 1)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)
    precision = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return total_loss / max(n, 1), acc, f1, precision, recall, cm


def trainable_params(m: torch.nn.Module):
    """Return only parameters that will be updated."""
    return [p for p in m.parameters() if p.requires_grad]


def main():
    set_seed(42)
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = HatefulMemesDataset(
        args.data_root, split="train",
        tokenizer=tokenizer, max_len=args.max_len, image_size=args.image_size
    )
    dev_ds = HatefulMemesDataset(
        args.data_root, split="dev",
        tokenizer=tokenizer, max_len=args.max_len, image_size=args.image_size
    )

    collate_fn = Collator(tokenizer, max_len=args.max_len)

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = FusionXAttnBinaryClassifier(
        freeze_image=args.freeze_image,
        freeze_text=args.freeze_text,
        # part-freeze knobs (recommended for this dataset)
        unfreeze_resnet_last_stage=args.unfreeze_resnet_last_stage,
        unfreeze_bert_last_n_layers=args.unfreeze_bert_last_n_layers,
    ).to(args.device)

    # ---- class-weighted loss ----
    class_weights = compute_class_weights_from_dataset(train_ds, args.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # ---- grouped learning rates ----
    encoder_lr = 1e-5
    fusion_lr = 1e-4       # cross-attn + projections + norms + ffn
    classifier_lr = 2e-4   # head

    # IMPORTANT: include fusion modules; only trainable params
    param_groups = []

    img_params = trainable_params(model.image_encoder)
    txt_params = trainable_params(model.text_encoder)

    if img_params:
        param_groups.append({"params": img_params, "lr": encoder_lr})
    if txt_params:
        param_groups.append({"params": txt_params, "lr": encoder_lr})

    # fusion modules (these were missing before!)
    param_groups.append({"params": trainable_params(model.img_proj), "lr": fusion_lr})
    param_groups.append({"params": trainable_params(model.cross_attn), "lr": fusion_lr})
    param_groups.append({"params": trainable_params(model.attn_ln), "lr": fusion_lr})
    param_groups.append({"params": trainable_params(model.ffn), "lr": fusion_lr})
    param_groups.append({"params": trainable_params(model.ffn_ln), "lr": fusion_lr})

    # classifier
    param_groups.append({"params": trainable_params(model.classifier), "lr": classifier_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    print(f"[lr groups] encoder_lr={encoder_lr} fusion_lr={fusion_lr} classifier_lr={classifier_lr}")
    print(
        "[trainable #params] "
        f"img={sum(p.numel() for p in img_params):,} "
        f"txt={sum(p.numel() for p in txt_params):,} "
        f"fusion={sum(p.numel() for p in trainable_params(model.img_proj)) + sum(p.numel() for p in trainable_params(model.cross_attn)) + sum(p.numel() for p in trainable_params(model.ffn)):,} "
        f"classifier={sum(p.numel() for p in trainable_params(model.classifier)):,}"
    )

    best_score = -1.0
    best_path = os.path.join(args.save_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        total_loss, total_correct, n = 0.0, 0, 0

        for batch in pbar:
            images = batch["images"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images, input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            bs = labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * bs
            n += bs

            pbar.set_postfix(loss=total_loss / max(n, 1), acc=total_correct / max(n, 1))

        dev_loss, dev_acc, dev_f1, dev_prec, dev_rec, dev_cm = evaluate(
            model, dev_loader, args.device, loss_fn
        )

        print(
            f"[epoch {epoch}] "
            f"dev_loss={dev_loss:.4f} "
            f"dev_acc={dev_acc:.4f} "
            f"dev_f1={dev_f1:.4f} "
            f"precision={dev_prec:.4f} "
            f"recall={dev_rec:.4f}"
        )
        print("confusion_matrix [[TN, FP], [FN, TP]]:")
        print(dev_cm)

        if dev_f1 > best_score:
            best_score = dev_f1
            torch.save(
                {"model": model.state_dict(), "best_f1": best_score, "args": vars(args)},
                best_path
            )
            print(f"[saved] {best_path} (best_f1={best_score:.4f})")

    print(f"Done. Best dev F1 = {best_score:.4f}")


if __name__ == "__main__":
    main()