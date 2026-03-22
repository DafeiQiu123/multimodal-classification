# models/fusion_model.py
import torch
import torch.nn as nn
import timm
from transformers import AutoModel
import torch.nn.functional as F


class FusionBinaryClassifier(nn.Module):
    """
    Image encoder (ResNet18) + Text encoder (DistilBERT)
    -> L2 normalize -> concat -> MLP -> logits(2)
    """

    def __init__(
        self,
        image_backbone="resnet18",
        text_backbone="roberta-base",
        dropout=0.2,
        freeze_image=False,
        freeze_text=False,
        unfreeze_resnet_last_stage=False,   # unfreeze ResNet last stage (layer4)
        unfreeze_bert_last_n_layers=0,      # unfreeze last N layers of text encoder
        unfreeze_bert_layer_norm=True,      # also unfreeze final layer_norm if any layer is unfrozen
    ):
        super().__init__()

        # ---- Image model (timm) ----
        self.image_encoder = timm.create_model(
            image_backbone,
            pretrained=True,
            num_classes=0,       # remove classification head
            global_pool="avg",
        )
        img_dim = self.image_encoder.num_features

        # ---- Text model (transformers) ----
        self.text_encoder = AutoModel.from_pretrained(text_backbone)
        txt_dim = self.text_encoder.config.hidden_size  # DistilBERT: 768

        # --------- freezing / partial unfreezing ----------
        def freeze_module(m: nn.Module):
            for p in m.parameters():
                p.requires_grad = False

        def unfreeze_module(m: nn.Module):
            for p in m.parameters():
                p.requires_grad = True

        # 1) Apply full freeze if requested
        if freeze_image:
            freeze_module(self.image_encoder)
        if freeze_text:
            freeze_module(self.text_encoder)

        # 2) Partial unfreeze (only if not fully frozen)
        # ---- ResNet: unfreeze last stage ----
        if (not freeze_image) and unfreeze_resnet_last_stage:
            # Start from frozen baseline for stability
            freeze_module(self.image_encoder)

            # timm resnet usually has .layer4; if not, we raise a clear error
            if hasattr(self.image_encoder, "layer4"):
                unfreeze_module(self.image_encoder.layer4)
            else:
                raise AttributeError(
                    f"image_encoder ({type(self.image_encoder)}) has no attribute 'layer4'. "
                    f"Print model to find the last stage name for backbone='{image_backbone}'."
                )

        # ---- Text encoder: unfreeze last N layers ----
        if (not freeze_text) and (unfreeze_bert_last_n_layers and unfreeze_bert_last_n_layers > 0):
            freeze_module(self.text_encoder)

            # RoBERTa/BERT: encoder.layer  |  DistilBERT: transformer.layer
            if hasattr(self.text_encoder, "encoder") and hasattr(self.text_encoder.encoder, "layer"):
                layers = self.text_encoder.encoder.layer          # RoBERTa / BERT
                layer_norm_parent = None                          # RoBERTa has no top-level LN
            elif hasattr(self.text_encoder, "transformer") and hasattr(self.text_encoder.transformer, "layer"):
                layers = self.text_encoder.transformer.layer      # DistilBERT
                layer_norm_parent = self.text_encoder.transformer
            else:
                raise AttributeError(
                    f"text_encoder ({type(self.text_encoder)}) exposes neither encoder.layer "
                    f"nor transformer.layer. Backbone='{text_backbone}' is unsupported."
                )

            n_total = len(layers)
            n = max(1, min(int(unfreeze_bert_last_n_layers), n_total))
            for layer in layers[-n:]:
                unfreeze_module(layer)

            if unfreeze_bert_layer_norm and layer_norm_parent is not None:
                if hasattr(layer_norm_parent, "layer_norm"):
                    unfreeze_module(layer_norm_parent.layer_norm)

        # --------- classifier head ----------
        fused_dim = img_dim + txt_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)  # binary -> 2 logits
        )

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_encoder(images)  # [B, img_dim]
        img_feat = F.normalize(img_feat, p=2, dim=1)     # L2 norm

        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = out.last_hidden_state[:, 0, :]        # [B, txt_dim] (CLS token)
        txt_feat = F.normalize(txt_feat, p=2, dim=1)     # L2 norm

        fused = torch.cat([img_feat, txt_feat], dim=1)   # [B, fused_dim]
        logits = self.classifier(fused)                  # [B, 2]
        return logits