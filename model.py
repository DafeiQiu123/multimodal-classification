# models/fusion_model.py
import torch
import torch.nn as nn
import timm
from transformers import AutoModel
import torch.nn.functional as F

class FusionBinaryClassifier(nn.Module):
    """
    Image encoder (ResNet18) + Text encoder (DistilBERT)
    -> concat features -> MLP -> logits(2)
    """
    def __init__(
        self,
        image_backbone="resnet18",
        text_backbone="distilbert-base-uncased",
        dropout=0.2,
        freeze_image=False,
        freeze_text=False,
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

        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        fused_dim = img_dim + txt_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)  # binary -> 2 logits
        )

    def forward(self, images, input_ids, attention_mask):
        img_feat = self.image_encoder(images)  # [B, img_dim]
        #  normalize!
        img_feat = F.normalize(img_feat, p=2, dim=1)

        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_feat = out.last_hidden_state[:, 0, :]  # [B, txt_dim] (CLS token)
        #  normalize!
        txt_feat = F.normalize(txt_feat, p=2, dim=1)
        
        fused = torch.cat([img_feat, txt_feat], dim=1)  # [B, fused_dim]
        logits = self.classifier(fused)                 # [B, 2]
        return logits