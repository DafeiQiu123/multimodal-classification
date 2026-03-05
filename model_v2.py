# models/fusion_model_xattn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoModel


class FusionXAttnBinaryClassifier(nn.Module):
    """
    Cross-attention fusion (方案A):
    - Image encoder: ResNet18 feature map -> tokens [B, HW, C]
    - Text encoder: DistilBERT tokens [B, T, 768]
    - Cross-attn: Q=text, K/V=image
    - Pool: masked mean on fused text tokens
    - Head: MLP -> logits(2)
    """

    def __init__(
        self,
        image_backbone="resnet18",
        text_backbone="distilbert-base-uncased",
        dropout=0.2,
        num_heads=8,
        # partial-freeze knobs (optional, keep same style)
        freeze_image=False,
        freeze_text=False,
        unfreeze_resnet_last_stage=False,
        unfreeze_bert_last_n_layers=0,
    ):
        super().__init__()

        # -------- Image encoder: get feature map (B,C,H,W) --------
        # features_only=True makes it easy to get intermediate feature maps
        self.image_encoder = timm.create_model(
            image_backbone,
            pretrained=True,
            features_only=True,
            out_indices=(-1,),   # last stage feature map
        )
        # timm FeaturesListNet provides feature_info with channels
        img_c = self.image_encoder.feature_info.channels()[-1]

        # -------- Text encoder: token features (B,T,768) --------
        self.text_encoder = AutoModel.from_pretrained(text_backbone)
        txt_dim = self.text_encoder.config.hidden_size  # DistilBERT: 768

        # Project image token dim -> txt_dim so attention works
        self.img_proj = nn.Linear(img_c, txt_dim)

        # Cross-attention (batch_first=True so input is [B, L, D])
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=txt_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Small FFN + LayerNorm (Transformer-style) to stabilize
        self.attn_ln = nn.LayerNorm(txt_dim)
        self.ffn = nn.Sequential(
            nn.Linear(txt_dim, txt_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(txt_dim * 4, txt_dim),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(txt_dim)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(txt_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2),
        )

        # -------- freezing / partial unfreezing --------
        def freeze_module(m: nn.Module):
            for p in m.parameters():
                p.requires_grad = False

        def unfreeze_module(m: nn.Module):
            for p in m.parameters():
                p.requires_grad = True

        if freeze_image:
            freeze_module(self.image_encoder)
        if freeze_text:
            freeze_module(self.text_encoder)

        # ResNet partial unfreeze: last stage only (layer4 in classic resnet)
        # With features_only wrapper, the underlying original model is typically at .model
        if (not freeze_image) and unfreeze_resnet_last_stage:
            freeze_module(self.image_encoder)
            # try to reach underlying model
            base = getattr(self.image_encoder, "model", None)
            if base is not None and hasattr(base, "layer4"):
                unfreeze_module(base.layer4)
            else:
                # fallback: unfreeze entire last feature block params if no layer4 exposed
                # (still partial-ish)
                for name, p in self.image_encoder.named_parameters():
                    if "layer4" in name:
                        p.requires_grad = True

        # DistilBERT partial unfreeze: last N layers
        if (not freeze_text) and unfreeze_bert_last_n_layers and unfreeze_bert_last_n_layers > 0:
            freeze_module(self.text_encoder)
            if hasattr(self.text_encoder, "transformer") and hasattr(self.text_encoder.transformer, "layer"):
                layers = self.text_encoder.transformer.layer
                n_total = len(layers)
                n = max(1, min(int(unfreeze_bert_last_n_layers), n_total))
                for layer in layers[-n:]:
                    unfreeze_module(layer)
                # also unfreeze final layer norm if present
                if hasattr(self.text_encoder.transformer, "layer_norm"):
                    unfreeze_module(self.text_encoder.transformer.layer_norm)
            else:
                raise AttributeError("text_encoder is not DistilBERT-like (missing transformer.layer).")

    def _masked_mean_pool(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        attention_mask: [B, T] with 1 for tokens, 0 for padding
        """
        mask = attention_mask.unsqueeze(-1).type_as(x)  # [B,T,1]
        x = x * mask
        denom = mask.sum(dim=1).clamp(min=1.0)          # [B,1]
        return x.sum(dim=1) / denom                     # [B,D]

    def forward(self, images, input_ids, attention_mask):
        # ---- image tokens ----
        feats = self.image_encoder(images)[0]  # last feature map: [B,C,H,W]
        B, C, H, W = feats.shape
        img_tokens = feats.flatten(2).transpose(1, 2)   # [B, HW, C]
        img_tokens = self.img_proj(img_tokens)          # [B, HW, D]
        img_tokens = F.normalize(img_tokens, p=2, dim=-1)  # optional, helps scale

        # ---- text tokens ----
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_tokens = out.last_hidden_state              # [B, T, D]
        txt_tokens = F.normalize(txt_tokens, p=2, dim=-1)  # optional

        # ---- cross-attention: Q=text, K/V=image ----
        attn_out, _ = self.cross_attn(
            query=txt_tokens,
            key=img_tokens,
            value=img_tokens,
            need_weights=False,
        )
        # Residual + LN
        x = self.attn_ln(txt_tokens + attn_out)

        # ---- FFN block ----
        ffn_out = self.ffn(x)
        x = self.ffn_ln(x + ffn_out)

        # ---- pool & classify ----
        pooled = self._masked_mean_pool(x, attention_mask)  # [B,D]
        logits = self.classifier(pooled)                    # [B,2]
        return logits