import torch
import torch.nn as nn
import timm
from transformers import AutoModel


class FusionXAttnBinaryClassifier(nn.Module):
    """
    Multimodal classifier with Cross Attention
    Image -> tokens
    Text -> tokens
    Text attends to image tokens
    """

    def __init__(
        self,
        image_backbone="resnet18",
        text_backbone="distilbert-base-uncased",
        num_heads=8,
        dropout=0.3,
        freeze_image=False,
        freeze_text=False,
        unfreeze_resnet_last_stage=False,
        unfreeze_bert_last_n_layers=0,
    ):
        super().__init__()

        # ---------------- IMAGE ENCODER ----------------
        self.image_encoder = timm.create_model(
            image_backbone,
            pretrained=True,
            features_only=True,
            out_indices=(-1,)
        )

        img_dim = self.image_encoder.feature_info.channels()[-1]

        # ---------------- TEXT ENCODER ----------------
        self.text_encoder = AutoModel.from_pretrained(text_backbone)

        txt_dim = self.text_encoder.config.hidden_size

        # ---------------- FUSION ----------------
        self.img_proj = nn.Linear(img_dim, txt_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=txt_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.attn_ln = nn.LayerNorm(txt_dim)
        self.ffn_ln = nn.LayerNorm(txt_dim)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(txt_dim, txt_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(txt_dim * 4, txt_dim)
        )

        # ---------------- CLASSIFIER ----------------
        self.classifier = nn.Sequential(
            nn.Linear(txt_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 2)
        )

        # ---------------- FREEZE ----------------
        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        # partial unfreeze image
        if unfreeze_resnet_last_stage:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

            for name, p in self.image_encoder.named_parameters():
                if "layer4" in name:
                    p.requires_grad = True

        # partial unfreeze text
        if unfreeze_bert_last_n_layers > 0:

            for p in self.text_encoder.parameters():
                p.requires_grad = False

            layers = self.text_encoder.transformer.layer
            for layer in layers[-unfreeze_bert_last_n_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def masked_mean_pool(self, x, mask):

        mask = mask.unsqueeze(-1)
        x = x * mask

        return x.sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, images, input_ids, attention_mask):

        # -------- IMAGE TOKENS --------
        feat = self.image_encoder(images)[0]

        img_tokens = feat.flatten(2).transpose(1, 2)
        img_tokens = self.img_proj(img_tokens)

        # -------- TEXT TOKENS --------
        txt = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        txt_tokens = txt.last_hidden_state

        # -------- CROSS ATTENTION --------
        attn_out, _ = self.cross_attn(
            query=txt_tokens,
            key=img_tokens,
            value=img_tokens
        )

        attn_out = self.dropout(attn_out)

        x = self.attn_ln(txt_tokens + attn_out)

        # -------- FFN --------
        ffn_out = self.ffn(x)

        ffn_out = self.dropout(ffn_out)

        x = self.ffn_ln(x + ffn_out)

        # -------- POOL --------
        pooled = self.masked_mean_pool(x, attention_mask)

        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)

        return logits