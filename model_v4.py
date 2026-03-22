import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel


class HatefulMemeFusionV4(nn.Module):
    """
    Backbone : CLIP-Large (clip-vit-large-patch14)  vis_dim=1024  txt_dim=768
    Fusion   : Bidirectional cross-attention
               - Text->Image (t2i): text tokens attend to image patches -> masked mean pool
               - Image->Text (i2t): image patches attend to text tokens -> CLS token pool
               concat(t2i_pooled, i2t_pooled) -> MLP -> logits(2)
    """

    def __init__(
        self,
        clip_backbone="openai/clip-vit-large-patch14",
        shared_dim=256,
        num_heads=8,
        dropout=0.3,
        freeze_vision=True,
        freeze_text=True,
        unfreeze_vision_last_n_layers=0,
        unfreeze_text_last_n_layers=0,
    ):
        super().__init__()

        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_backbone)
        self.text_encoder   = CLIPTextModel.from_pretrained(clip_backbone)

        vis_dim = self.vision_encoder.config.hidden_size   # 1024 for ViT-L/14
        txt_dim = self.text_encoder.config.hidden_size     # 768  for ViT-L/14

        def freeze(m):
            for p in m.parameters(): p.requires_grad = False
        def unfreeze(m):
            for p in m.parameters(): p.requires_grad = True

        if freeze_vision: freeze(self.vision_encoder)
        if freeze_text:   freeze(self.text_encoder)

        if unfreeze_vision_last_n_layers > 0:
            freeze(self.vision_encoder)
            vis_layers = self.vision_encoder.vision_model.encoder.layers
            n = max(1, min(unfreeze_vision_last_n_layers, len(vis_layers)))
            for layer in vis_layers[-n:]: unfreeze(layer)
            if hasattr(self.vision_encoder.vision_model, "post_layernorm"):
                unfreeze(self.vision_encoder.vision_model.post_layernorm)

        if unfreeze_text_last_n_layers > 0:
            freeze(self.text_encoder)
            txt_layers = self.text_encoder.text_model.encoder.layers
            n = max(1, min(unfreeze_text_last_n_layers, len(txt_layers)))
            for layer in txt_layers[-n:]: unfreeze(layer)
            if hasattr(self.text_encoder.text_model, "final_layer_norm"):
                unfreeze(self.text_encoder.text_model.final_layer_norm)

        self.vis_proj = nn.Linear(vis_dim, shared_dim)
        self.txt_proj = nn.Linear(txt_dim, shared_dim)
        self.dropout  = nn.Dropout(dropout)

        # Text -> Image cross-attention
        self.t2i_attn    = nn.MultiheadAttention(shared_dim, num_heads, dropout=dropout, batch_first=True)
        self.t2i_attn_ln = nn.LayerNorm(shared_dim)
        self.t2i_ffn     = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 4, shared_dim),
        )
        self.t2i_ffn_ln  = nn.LayerNorm(shared_dim)

        # Image -> Text cross-attention
        self.i2t_attn    = nn.MultiheadAttention(shared_dim, num_heads, dropout=dropout, batch_first=True)
        self.i2t_attn_ln = nn.LayerNorm(shared_dim)
        self.i2t_ffn     = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 4, shared_dim),
        )
        self.i2t_ffn_ln  = nn.LayerNorm(shared_dim)

        # Classifier: concat(t2i_pooled, i2t_pooled) -> [B, shared_dim*2]
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim * 2),
            nn.Linear(shared_dim * 2, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 2),
        )

    def masked_mean_pool(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, pixel_values, input_ids, attention_mask):
        vis_out = self.vision_encoder(pixel_values=pixel_values)
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)

        vis_tokens = self.vis_proj(vis_out.last_hidden_state)   # [B, num_patches+1, D]
        txt_tokens = self.txt_proj(txt_out.last_hidden_state)   # [B, seq_len, D]

        # Text -> Image
        t2i_out, _ = self.t2i_attn(query=txt_tokens, key=vis_tokens, value=vis_tokens)
        t2i = self.t2i_attn_ln(txt_tokens + self.dropout(t2i_out))
        t2i = self.t2i_ffn_ln(t2i + self.dropout(self.t2i_ffn(t2i)))
        t2i_pooled = self.masked_mean_pool(t2i, attention_mask)     # [B, D]

        # Image -> Text
        i2t_out, _ = self.i2t_attn(query=vis_tokens, key=txt_tokens, value=txt_tokens)
        i2t = self.i2t_attn_ln(vis_tokens + self.dropout(i2t_out))
        i2t = self.i2t_ffn_ln(i2t + self.dropout(self.i2t_ffn(i2t)))
        i2t_pooled = i2t[:, 0, :]                                   # CLS token [B, D]

        fused = torch.cat([t2i_pooled, i2t_pooled], dim=1)          # [B, D*2]
        return self.classifier(fused)
