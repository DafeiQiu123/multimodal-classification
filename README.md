# Hateful Meme Detection — Model Development Report

## 1. Task Overview

Binary classification on the [Hateful Memes dataset](https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set/) (Facebook AI, 2020). Each sample is a meme: an image with overlaid text. The label indicates whether the meme is hateful (1) or not (0).

The challenge is that neither image nor text alone is sufficient — the hateful meaning often emerges from the **combination** of both modalities (e.g., an innocent image paired with racist text, or vice versa).

**Primary metric**: F1 score (binary, positive class = hateful).

---

## 2. Dataset Statistics

| Split | Total | Negative (0) | Positive (1) | Pos Ratio |
| ----- | ----- | ------------ | ------------ | --------- |
| Train | 8,500 | 5,450        | 3,050        | 35.9%     |
| Dev   | 500   | 250          | 250          | 50.0%     |

**Key observation**: The train/dev distribution mismatch (35.9% vs 50.0% positive) causes the model to be biased toward the negative class if trained naively. We address this with class-weighted loss throughout all versions.

Class weights used:

```
w0 = N / (2 * n0) = 0.7798
w1 = N / (2 * n1) = 1.3934
```

---

## 3. Model Evolution

### 3.1 V1 — Late Fusion Baseline (ResNet18 + DistilBERT)

**Architecture**

```
Image  → ResNet18 (timm, pretrained) → global avg pool → L2 norm → [B, 512]
Text   → DistilBERT → CLS token     → L2 norm         → [B, 768]
Concat → [B, 1280] → Linear(512) → ReLU → Dropout → Linear(2)
```

**Training**

- Grouped learning rates: encoder `1e-5`, classifier `2e-4`
- Partial freeze: ResNet layer4 + DistilBERT last 1 layer unfrozen
- Class-weighted CrossEntropyLoss
- 5 epochs

**Result**: Dev Acc = 0.622, **Dev F1 = 0.585**

**Limitation**: Simple concatenation treats the two modalities as independent — no cross-modal interaction is modelled.

---

### 3.2 V2 — Cross-Attention Fusion (ResNet18 + DistilBERT)

**Architecture change**: Replace concat with a Transformer-style cross-attention block.

```
Image  → ResNet18 → feature map → flatten → [B, HW, 512]
Text   → DistilBERT → token sequence → [B, seq_len, 768]
                          |
         img_proj: Linear(512 → 768)
                          |
         Cross-Attention: query=txt_tokens, key/value=img_tokens
         + Residual + LayerNorm + FFN + LayerNorm
                          |
         Masked mean pool over text positions
                          |
         Linear(512) → ReLU → Dropout → Linear(2)
```

**Key insight**: Text tokens attend to image patch tokens — the model learns _which image regions are relevant given the text context_.

**Training**: Same grouped LRs, added `fusion_lr=1e-4` for the cross-attention module.

**Result**: Improved cross-modal reasoning over V1.

**Limitation**: Both backbones (ResNet18, DistilBERT) were trained independently on separate tasks — their feature spaces are not aligned. The fusion module must bridge a large semantic gap from scratch.

---

### 3.3 V3 — CLIP Backbone + Training Improvements

**Core change**: Replace ResNet18 + DistilBERT with CLIP's separate vision and text encoders (`clip-vit-base-patch32`). CLIP is pre-trained with a contrastive objective that **aligns image and text into a shared semantic space**, making the fusion task significantly easier.

**Architecture**

```
Image  → CLIPVisionModel → last_hidden_state → [B, 50, 768]
Text   → CLIPTextModel  → last_hidden_state → [B, seq_len, 512]
                              |
         vis_proj: Linear(768 → 512)
                              |
         Cross-Attention: query=txt_tokens, key/value=vis_tokens
         + Residual + LayerNorm + FFN(4x) + LayerNorm
                              |
         Masked mean pool (text positions)
                              |
         Linear(256) → ReLU → Dropout → Linear(2)
```

**Training improvements introduced in V3**

| Technique                   | Details                                                                                                               |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Two-stage training          | Stage 1: encoders fully frozen (only fusion + classifier train). Stage 2: unfreeze last N encoder layers with low LR. |
| Cosine LR schedule + warmup | `get_cosine_schedule_with_warmup`, warmup_ratio=0.1                                                                   |
| Gradient clipping           | `max_norm=1.0` prevents fusion instability                                                                            |
| Label smoothing             | `label_smoothing=0.1` reduces overconfident predictions                                                               |
| Data augmentation           | Random horizontal flip, color jitter, grayscale, rotation (PIL-space, before CLIPProcessor)                           |
| Gradient-grouped optimizer  | Separate LRs for encoder (`5e-7`), fusion (`1e-4`), classifier (`2e-4`)                                               |

**Why two-stage training matters**: With only 8,500 training samples, fine-tuning all parameters simultaneously causes severe overfitting. Freezing the encoders in stage 1 lets the fusion head learn a stable mapping first, then stage 2 fine-tunes the encoders from a good initialization.

**Result**: **Best Dev F1 = 0.6641** (epoch 11, stage 2)

---

### 3.4 V4 — Bidirectional Cross-Attention + CLIP-Large + R-Drop

**Backbone upgrade**: `clip-vit-base-patch32` → `clip-vit-large-patch14`

| Backbone | Vision dim | Text dim | Vision patches | Parameters |
| -------- | ---------- | -------- | -------------- | ---------- |
| ViT-B/32 | 768        | 512      | 49 + 1 = 50    | ~150M      |
| ViT-L/14 | 1024       | 768      | 256 + 1 = 257  | ~430M      |

ViT-L/14 extracts significantly richer visual features (4x more patches, 1.3x wider hidden dim).

**Architecture change: Bidirectional cross-attention**

V3 only modelled text-attends-to-image. V4 adds the reverse direction:

```
Image  → CLIPVisionModel → [B, 257, 1024] → vis_proj → [B, 257, 256]
Text   → CLIPTextModel  → [B, seq_len, 768] → txt_proj → [B, seq_len, 256]

Text -> Image (t2i):
    Cross-Attn: query=txt_tokens, key/value=vis_tokens
    + Residual + LayerNorm + FFN(4x) + LayerNorm
    Masked mean pool → t2i_pooled: [B, 256]

Image -> Text (i2t):
    Cross-Attn: query=vis_tokens, key/value=txt_tokens
    + Residual + LayerNorm + FFN(4x) + LayerNorm
    CLS token → i2t_pooled: [B, 256]

concat([t2i_pooled, i2t_pooled]) → [B, 512]
LayerNorm → Linear(256) → GELU → Dropout → Linear(2)
```

**Why bidirectional?**

- **t2i** (text→image): "Given this text, which image regions are relevant?" — grounds language in visual context
- **i2t** (image→text): "Given these image patches, what text concepts are relevant?" — grounds vision in linguistic context

Hateful memes require both: the model must understand _what the image shows_ in the context of the text, and _what the text means_ in the context of the image.

**R-Drop regularization** (stage 2 only)

Standard dropout randomly silences neurons, but two forward passes with different masks can produce inconsistent predictions — especially harmful for small datasets. R-Drop forces consistency:

```
Forward pass 1: logits_1 = model(x)   # dropout mask A
Forward pass 2: logits_2 = model(x)   # dropout mask B

loss = 0.5 * (CE(logits_1, y) + CE(logits_2, y))
     + alpha * KL_symmetric(p1 || p2)
```

R-Drop is applied in stage 2 only. In stage 1 the fusion module is randomly initialized, so the two dropout masks would produce very different outputs, making the KL penalty unstable.

**Early stopping** (patience=5 per stage)

Automatically halts each stage when dev F1 stops improving, preventing the characteristic "recall collapse" seen in earlier runs where precision climbs but recall drops.

**Result**: **Best Dev F1 = 0.6695** (epoch 5, stage 1)

---

## 4. Results Summary

| Version | Backbone              | Fusion            | Best Dev F1 | Dev Acc |
| ------- | --------------------- | ----------------- | ----------- | ------- |
| V1      | ResNet18 + DistilBERT | Concat            | 0.585       | 0.622   |
| V2      | ResNet18 + DistilBERT | Cross-Attn (t2i)  | 0.630       | 0.598   |
| V3      | CLIP ViT-B/32         | Cross-Attn (t2i)  | **0.6641**  | 0.654   |
| V4      | CLIP ViT-L/14         | Bidir. Cross-Attn | **0.6695**  | 0.684   |

---

## 5. Key Lessons

**1. Backbone alignment matters most.**
Switching from ResNet18 + DistilBERT to CLIP (V1→V3) gave the largest single improvement (+0.079 F1). CLIP's contrastive pretraining reduces the semantic gap between image and text features before any fusion occurs.

**2. Overfitting is the dominant challenge on this dataset.**
With only 8,500 training samples, even 3.5M fusion parameters overfit in 1–2 epochs. Every design decision in V3 and V4 was driven by reducing overfitting:

- Two-stage training (freeze encoders first)
- Label smoothing
- Gradient clipping
- Early stopping
- R-Drop

**3. Two-stage training is essential for frozen CLIP.**
Stage 1 (frozen encoders) lets the fusion module learn a stable projection from CLIP's frozen feature space. Stage 2 (partial encoder unfreeze) then performs careful domain adaptation with very low LR (`5e-7`). Reversing this order — or not separating the stages — consistently produced worse results.

**4. Recall collapse is a key failure mode.**
Across all runs, models that continued training past the optimum showed a consistent pattern: precision increased but recall dropped sharply. This reflects the model becoming increasingly conservative (predicting "not hateful" for uncertain examples). Early stopping prevents this.

**5. Larger backbone + smaller fusion head is better than the reverse.**
V4 uses a smaller fusion module (shared_dim=256, ~2M params) on top of a much larger backbone (ViT-L/14, ~430M frozen params) compared to V3's larger fusion on a smaller backbone. Trusting the pretrained representations and keeping the task-specific head small is the right inductive bias for low-data regimes.
