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
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="../data/hateful_meme")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--encoder_lr", type=float, default=1e-5)
    parser.add_argument("--fusion_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_lr", type=float, default=2e-4)

    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--freeze_image", action="store_true")
    parser.add_argument("--freeze_text", action="store_true")

    parser.add_argument("--unfreeze_resnet_last_stage", action="store_true")
    parser.add_argument("--unfreeze_bert_last_n_layers", type=int, default=1)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


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

    preds = []
    labels = []

    total_loss = 0.0
    total_correct = 0
    n = 0

    for batch in loader:
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(images, input_ids, attention_mask)
        loss = loss_fn(logits, y)

        pred = logits.argmax(dim=1)

        preds.extend(pred.detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())

        total_loss += loss.item() * y.size(0)
        total_correct += (pred == y).sum().item()
        n += y.size(0)

    acc = total_correct / n
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    cm = confusion_matrix(labels, preds)

    return total_loss / n, acc, f1, precision, recall, cm


def trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs("outputs", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = HatefulMemesDataset(
        args.data_root, "train", tokenizer, args.max_len, args.image_size
    )
    dev_ds = HatefulMemesDataset(
        args.data_root, "dev", tokenizer, args.max_len, args.image_size
    )

    collator = Collator(tokenizer, args.max_len)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        generator=g,
        pin_memory=torch.cuda.is_available(),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = FusionXAttnBinaryClassifier(
        freeze_image=args.freeze_image,
        freeze_text=args.freeze_text,
        unfreeze_resnet_last_stage=args.unfreeze_resnet_last_stage,
        unfreeze_bert_last_n_layers=args.unfreeze_bert_last_n_layers
    ).to(args.device)

    class_weights = compute_class_weights(train_ds, args.device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    img_params = trainable_params(model.image_encoder)
    txt_params = trainable_params(model.text_encoder)
    fusion_params = (
        trainable_params(model.img_proj)
        + trainable_params(model.cross_attn)
        + trainable_params(model.attn_ln)
        + trainable_params(model.ffn)
        + trainable_params(model.ffn_ln)
    )
    cls_params = trainable_params(model.classifier)

    optimizer = torch.optim.AdamW(
        [
            {"params": img_params, "lr": args.encoder_lr},
            {"params": txt_params, "lr": args.encoder_lr},
            {"params": fusion_params, "lr": args.fusion_lr},
            {"params": cls_params, "lr": args.classifier_lr},
        ],
        weight_decay=args.weight_decay,
    )

    print(
        f"[lr groups] encoder_lr={args.encoder_lr} fusion_lr={args.fusion_lr} classifier_lr={args.classifier_lr}"
    )
    print(
        f"[trainable #params] "
        f"img={sum(p.numel() for p in img_params):,} "
        f"txt={sum(p.numel() for p in txt_params):,} "
        f"fusion={sum(p.numel() for p in fusion_params):,} "
        f"classifier={sum(p.numel() for p in cls_params):,}"
    )

    best_f1 = -1.0

    for epoch in range(args.epochs):
        model.train()

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        total_loss = 0.0
        total_correct = 0
        total_n = 0

        for batch in pbar:
            images = batch["images"].to(args.device)
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            labels = batch["labels"].to(args.device)

            optimizer.zero_grad()

            logits = model(images, input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            bs = labels.size(0)

            total_loss += loss.item() * bs
            total_correct += (preds == labels).sum().item()
            total_n += bs

            pbar.set_postfix(
                loss=total_loss / max(total_n, 1),
                acc=total_correct / max(total_n, 1),
            )

        dev_loss, dev_acc, dev_f1, prec, rec, cm = evaluate(
            model, dev_loader, args.device, loss_fn
        )

        print(
            f"[epoch {epoch+1}] "
            f"dev_loss={dev_loss:.4f} "
            f"dev_acc={dev_acc:.4f} "
            f"dev_f1={dev_f1:.4f} "
            f"precision={prec:.4f} "
            f"recall={rec:.4f}"
        )

        print("confusion_matrix [[TN, FP], [FN, TP]]")
        print(cm)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.state_dict(), "outputs/best.pt")
            print(f"[saved] best model (f1={best_f1:.4f})")

    print(f"Training finished. Best dev F1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()