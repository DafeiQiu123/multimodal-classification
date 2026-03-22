import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel


class CLIPFusionClassifier(nn.Module):
    """
    CLIP vision encoder + CLIP text encoder (kept separate, not the joint CLIPModel)
    -> project vision tokens to text dim
    -> cross-attention (text attends to image patches)
    -> masked mean pool -> MLP -> logits(2)
    """

    def __init__(
        self,
        clip_backbone="openai/clip-vit-base-patch32",
        num_heads=8,
        dropout=0.3,
        freeze_vision=True,
        freeze_text=True,
        unfreeze_vision_last_n_layers=0,
        unfreeze_text_last_n_layers=0,
    ):
        super().__init__()

        # ---------------- ENCODERS ----------------
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_backbone)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_backbone)

        vis_dim = self.vision_encoder.config.hidden_size   # 768 for ViT-B/32
        txt_dim = self.text_encoder.config.hidden_size     # 512 for ViT-B/32

        # ---------------- FREEZE ----------------
        def freeze(m):
            for p in m.parameters():
                p.requires_grad = False

        def unfreeze(m):
            for p in m.parameters():
                p.requires_grad = True

        if freeze_vision:
            freeze(self.vision_encoder)
        if freeze_text:
            freeze(self.text_encoder)

        if unfreeze_vision_last_n_layers > 0:
            freeze(self.vision_encoder)
            vis_layers = self.vision_encoder.vision_model.encoder.layers
            n = max(1, min(unfreeze_vision_last_n_layers, len(vis_layers)))
            for layer in vis_layers[-n:]:
                unfreeze(layer)
            if hasattr(self.vision_encoder.vision_model, "post_layernorm"):
                unfreeze(self.vision_encoder.vision_model.post_layernorm)

        if unfreeze_text_last_n_layers > 0:
            freeze(self.text_encoder)
            txt_layers = self.text_encoder.text_model.encoder.layers
            n = max(1, min(unfreeze_text_last_n_layers, len(txt_layers)))
            for layer in txt_layers[-n:]:
                unfreeze(layer)
            if hasattr(self.text_encoder.text_model, "final_layer_norm"):
                unfreeze(self.text_encoder.text_model.final_layer_norm)

        # ---------------- FUSION ----------------
        self.vis_proj = nn.Linear(vis_dim, txt_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=txt_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn_ln = nn.LayerNorm(txt_dim)

        self.ffn = nn.Sequential(
            nn.Linear(txt_dim, txt_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(txt_dim * 4, txt_dim),
        )
        self.ffn_ln = nn.LayerNorm(txt_dim)

        self.dropout = nn.Dropout(dropout)

        # ---------------- CLASSIFIER ----------------
        self.classifier = nn.Sequential(
            nn.Linear(txt_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def masked_mean_pool(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, pixel_values, input_ids, attention_mask):
        # -------- VISION TOKENS --------
        vis_out = self.vision_encoder(pixel_values=pixel_values)
        vis_tokens = self.vis_proj(vis_out.last_hidden_state)       # [B, 50, txt_dim]

        # -------- TEXT TOKENS --------
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        txt_tokens = txt_out.last_hidden_state                      # [B, seq_len, txt_dim]

        # -------- CROSS ATTENTION (text queries image) --------
        attn_out, _ = self.cross_attn(query=txt_tokens, key=vis_tokens, value=vis_tokens)
        x = self.attn_ln(txt_tokens + self.dropout(attn_out))

        # -------- FFN --------
        x = self.ffn_ln(x + self.dropout(self.ffn(x)))

        # -------- POOL + CLASSIFY --------
        pooled = self.masked_mean_pool(x, attention_mask)
        return self.classifier(self.dropout(pooled))
