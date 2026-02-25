# train.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import f1_score
import numpy as np

from data_prepare import HatefulMemesDataset, Collator
from model import FusionBinaryClassifier

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=r"C:\Users\Administrator\Desktop\toxic\data\hateful_meme")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_len", type=int, default=64)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_dir", type=str, default="./outputs")
    p.add_argument("--freeze_image", action="store_true")
    p.add_argument("--freeze_text", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
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

    # binary F1 (positive label = 1)
    f1 = f1_score(y_true, y_pred, average="binary", pos_label=1)

    return total_loss / max(n, 1), acc, f1

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = HatefulMemesDataset(args.data_root, split="train",
                                   tokenizer=tokenizer, max_len=args.max_len, image_size=args.image_size)
    dev_ds = HatefulMemesDataset(args.data_root, split="dev",
                                 tokenizer=tokenizer, max_len=args.max_len, image_size=args.image_size)

    collate_fn = Collator(tokenizer, max_len=args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    model = FusionBinaryClassifier(
        freeze_image=args.freeze_image,
        freeze_text=args.freeze_text,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
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

            pbar.set_postfix(loss=total_loss/max(n,1), acc=total_correct/max(n,1))

        dev_loss, dev_acc, dev_f1 = evaluate(model, dev_loader, args.device)
        print(f"[epoch {epoch}] dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f} dev_f1={dev_f1:.4f}")

        if dev_acc > best_acc:
            best_acc = dev_acc
            torch.save({"model": model.state_dict(), "best_acc": best_acc, "args": vars(args)}, best_path)
            print(f"[saved] {best_path} (best_acc={best_acc:.4f})")

    print(f"Done. Best dev acc = {best_acc:.4f}")

if __name__ == "__main__":
    main()