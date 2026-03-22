import os
import json
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from model_v4 import HatefulMemeFusionV4


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",    type=str,   default="../data/hateful_meme")
    p.add_argument("--clip_backbone",type=str,   default="openai/clip-vit-large-patch14")
    p.add_argument("--shared_dim",   type=int,   default=256)
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--stage1_epochs",type=int,   default=10)
    p.add_argument("--encoder_lr",   type=float, default=5e-7)
    p.add_argument("--fusion_lr",    type=float, default=1e-4)
    p.add_argument("--classifier_lr",type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--label_smoothing",type=float,default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--rdrop_alpha",  type=float, default=0.5)
    p.add_argument("--unfreeze_vision_last_n_layers", type=int, default=1)
    p.add_argument("--unfreeze_text_last_n_layers",   type=int, default=1)
    p.add_argument("--patience",     type=int,   default=5)   # early stopping per stage
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--max_len",      type=int,   default=77)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


TRAIN_AUG = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomRotation(degrees=10),
])


class CLIPMemesDataset(Dataset):
    def __init__(self, data_root, split, augment=False):
        self.data_root = data_root
        self.augment   = augment
        self.items = []
        with open(os.path.join(data_root, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        image = Image.open(os.path.join(self.data_root, ex["img"])).convert("RGB")
        if self.augment:
            image = TRAIN_AUG(image)
        return {"image": image, "text": ex.get("text", ""),
                "label": int(ex.get("label", -1)), "id": ex.get("id", idx)}


class CLIPCollator:
    def __init__(self, processor, max_len=77):
        self.processor = processor
        self.max_len   = max_len

    def __call__(self, batch):
        enc = self.processor(
            images=[b["image"] for b in batch],
            text=[b["text"] for b in batch],
            return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_len,
        )
        return {
            "pixel_values":   enc["pixel_values"],
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        }


def compute_class_weights(dataset, device):
    labels = [x["label"] for x in dataset.items]
    n, n0, n1 = len(labels), labels.count(0), labels.count(1)
    w0, w1 = n / (2 * n0), n / (2 * n1)
    print(f"[class weights] n={n} n0={n0} n1={n1} -> w0={w0:.4f} w1={w1:.4f}")
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def rdrop_loss(logits1, logits2, labels, ce_fn, alpha):
    """CE on both passes + symmetric KL divergence (R-Drop regularization)."""
    ce = (ce_fn(logits1, labels) + ce_fn(logits2, labels)) * 0.5
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)
    kl = (F.kl_div(p1.log(), p2, reduction="batchmean") +
          F.kl_div(p2.log(), p1, reduction="batchmean")) * 0.5
    return ce + alpha * kl


@torch.no_grad()
def evaluate(model, loader, device, ce_fn):
    model.eval()
    preds, labels_all = [], []
    total_loss, total_correct, n = 0.0, 0, 0
    for batch in loader:
        pv  = batch["pixel_values"].to(device)
        ids = batch["input_ids"].to(device)
        msk = batch["attention_mask"].to(device)
        y   = batch["labels"].to(device)
        logits = model(pv, ids, msk)
        pred   = logits.argmax(dim=1)
        preds.extend(pred.cpu().numpy())
        labels_all.extend(y.cpu().numpy())
        total_loss    += ce_fn(logits, y).item() * y.size(0)
        total_correct += (pred == y).sum().item()
        n             += y.size(0)
    acc  = total_correct / n
    f1   = f1_score(labels_all, preds)
    prec = precision_score(labels_all, preds, zero_division=0)
    rec  = recall_score(labels_all, preds, zero_division=0)
    cm   = confusion_matrix(labels_all, preds)
    return total_loss / n, acc, f1, prec, rec, cm


def set_encoder_grad(model, unfreeze_vision_n=0, unfreeze_text_n=0):
    for p in model.vision_encoder.parameters(): p.requires_grad = False
    for p in model.text_encoder.parameters():   p.requires_grad = False
    if unfreeze_vision_n > 0:
        vis_layers = model.vision_encoder.vision_model.encoder.layers
        for layer in vis_layers[-unfreeze_vision_n:]:
            for p in layer.parameters(): p.requires_grad = True
        if hasattr(model.vision_encoder.vision_model, "post_layernorm"):
            for p in model.vision_encoder.vision_model.post_layernorm.parameters():
                p.requires_grad = True
    if unfreeze_text_n > 0:
        txt_layers = model.text_encoder.text_model.encoder.layers
        for layer in txt_layers[-unfreeze_text_n:]:
            for p in layer.parameters(): p.requires_grad = True
        if hasattr(model.text_encoder.text_model, "final_layer_norm"):
            for p in model.text_encoder.text_model.final_layer_norm.parameters():
                p.requires_grad = True


def trainable_params(m):
    return [p for p in m.parameters() if p.requires_grad]


def build_optimizer_and_scheduler(model, args, n_steps):
    vis_p = trainable_params(model.vision_encoder)
    txt_p = trainable_params(model.text_encoder)
    fus_p = (trainable_params(model.vis_proj)    + trainable_params(model.txt_proj)
           + trainable_params(model.t2i_attn)    + trainable_params(model.t2i_attn_ln)
           + trainable_params(model.t2i_ffn)     + trainable_params(model.t2i_ffn_ln)
           + trainable_params(model.i2t_attn)    + trainable_params(model.i2t_attn_ln)
           + trainable_params(model.i2t_ffn)     + trainable_params(model.i2t_ffn_ln))
    cls_p = trainable_params(model.classifier)

    groups = [{"params": fus_p, "lr": args.fusion_lr},
              {"params": cls_p, "lr": args.classifier_lr}]
    if vis_p: groups.insert(0, {"params": vis_p, "lr": args.encoder_lr})
    if txt_p: groups.insert(0, {"params": txt_p, "lr": args.encoder_lr})

    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sch = get_cosine_schedule_with_warmup(opt, int(n_steps * args.warmup_ratio), n_steps)
    print(f"[trainable] vision={sum(p.numel() for p in vis_p):,} "
          f"text={sum(p.numel() for p in txt_p):,} "
          f"fusion={sum(p.numel() for p in fus_p):,} "
          f"classifier={sum(p.numel() for p in cls_p):,}")
    return opt, sch


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs("outputs", exist_ok=True)

    processor = CLIPProcessor.from_pretrained(args.clip_backbone)
    collator  = CLIPCollator(processor, max_len=args.max_len)

    train_ds = CLIPMemesDataset(args.data_root, "train", augment=True)
    dev_ds   = CLIPMemesDataset(args.data_root, "dev",   augment=False)

    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collator, num_workers=args.num_workers,
                              generator=g, pin_memory=torch.cuda.is_available())
    dev_loader   = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collator, num_workers=args.num_workers,
                              pin_memory=torch.cuda.is_available())

    model = HatefulMemeFusionV4(
        clip_backbone=args.clip_backbone,
        shared_dim=args.shared_dim,
        freeze_vision=True,
        freeze_text=True,
    ).to(args.device)

    class_weights = compute_class_weights(train_ds, args.device)
    ce_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    stage2_epochs = args.epochs - args.stage1_epochs
    print(f"[two-stage] stage1={args.stage1_epochs} frozen | "
          f"stage2={stage2_epochs} unfreeze {args.unfreeze_vision_last_n_layers}v/"
          f"{args.unfreeze_text_last_n_layers}t layers")
    print(f"[R-Drop] alpha={args.rdrop_alpha} (stage 2 only)")

    best_f1 = -1.0
    no_improve = 0   # early-stopping counter (resets between stages)

    def run_epoch(epoch, total, opt, sch, use_rdrop):
        nonlocal best_f1, no_improve
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{total}")
        tot_loss, tot_correct, tot_n = 0.0, 0, 0

        for batch in pbar:
            pv     = batch["pixel_values"].to(args.device)
            ids    = batch["input_ids"].to(args.device)
            msk    = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            opt.zero_grad()
            if use_rdrop:
                l1 = model(pv, ids, msk)
                l2 = model(pv, ids, msk)
                loss   = rdrop_loss(l1, l2, labels, ce_fn, args.rdrop_alpha)
                logits = (l1 + l2) * 0.5
            else:
                logits = model(pv, ids, msk)
                loss   = ce_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            sch.step()

            pred = logits.argmax(dim=1)
            bs   = labels.size(0)
            tot_loss    += loss.item() * bs
            tot_correct += (pred == labels).sum().item()
            tot_n       += bs
            pbar.set_postfix(loss=tot_loss / max(tot_n, 1),
                             acc=tot_correct / max(tot_n, 1))

        dev_loss, dev_acc, dev_f1, prec, rec, cm = evaluate(model, dev_loader, args.device, ce_fn)
        print(f"[epoch {epoch}/{total}] dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} "
              f"dev_f1={dev_f1:.4f} precision={prec:.4f} recall={rec:.4f}")
        print("confusion_matrix [[TN, FP], [FN, TP]]")
        print(cm)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            no_improve = 0
            torch.save(model.state_dict(), "outputs/best_v4.pt")
            print(f"[saved] best_v4.pt (f1={best_f1:.4f})")
        else:
            no_improve += 1
            print(f"[no improve {no_improve}/{args.patience}]")

        return no_improve >= args.patience  # True = stop

    print("\n=== Stage 1: frozen encoders ===")
    set_encoder_grad(model, 0, 0)
    opt, sch = build_optimizer_and_scheduler(
        model, args, n_steps=args.stage1_epochs * len(train_loader))
    for epoch in range(1, args.stage1_epochs + 1):
        if run_epoch(epoch, args.epochs, opt, sch, use_rdrop=False):
            print(f"[early stop] stage 1 stopped at epoch {epoch}")
            break

    if stage2_epochs > 0:
        print("\n=== Stage 2: partial encoder unfreeze + R-Drop ===")
        # reload best checkpoint from stage 1 before fine-tuning encoders
        model.load_state_dict(torch.load("outputs/best_v4.pt", map_location=args.device))
        set_encoder_grad(model, args.unfreeze_vision_last_n_layers,
                                args.unfreeze_text_last_n_layers)
        no_improve = 0  # reset patience for stage 2
        opt, sch = build_optimizer_and_scheduler(
            model, args, n_steps=stage2_epochs * len(train_loader))
        for epoch in range(args.stage1_epochs + 1, args.epochs + 1):
            if run_epoch(epoch, args.epochs, opt, sch, use_rdrop=True):
                print(f"[early stop] stage 2 stopped at epoch {epoch}")
                break

    print(f"\nTraining finished. Best dev F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
