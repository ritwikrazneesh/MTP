from typing import List, Tuple
import torch
import torch.nn as nn
import open_clip

def build_remoteclip_via_openclip(model_name: str, checkpoint_path: str, device: str = "cuda"):
    """
    Per RemoteCLIP README: install open-clip-torch and load their checkpoint path
    as the `pretrained` argument. Example model_name: 'ViT-B-32'.
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=checkpoint_path,
        device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer, preprocess

class RemoteCLIPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def logit_scale(self):
        return self.model.logit_scale

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.encode_image(images)
        return feats / feats.norm(dim=-1, keepdim=True)

    def encode_text_from_tokens(self, token_embeddings: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # Largely mirrors OpenCLIP text forward, but accepts prebuilt token embeddings
        text = token_embeddings + self.model.positional_embedding.to(token_embeddings.dtype)
        context_length = text.shape[1]
        causal_mask = torch.full((context_length, context_length), float("-inf"), device=text.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        text = self.model.transformer(text, attn_mask=causal_mask)
        text = self.model.ln_final(text)
        lengths = attn_mask.sum(dim=1) - 1  # EOT positions
        x = text[torch.arange(text.shape[0], device=text.device), lengths]
        x = x @ self.model.text_projection
        return x / x.norm(dim=-1, keepdim=True)

class PromptLearner(nn.Module):
    def __init__(self, model, tokenizer, classnames: List[str], n_ctx: int = 8, template: str = "a satellite photo of a {}.", device: str = "cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classnames = [c.replace("_", " ") for c in classnames]
        self.n_ctx = n_ctx
        self.template = template

        prompts = [self.template.format(name) for name in self.classnames]
        tokenized = self.tokenizer(prompts)
        self.register_buffer("tokenized", tokenized)

        d = self.model.transformer.width
        # Prompt context as trainable parameter
        self.ctx = nn.Parameter(torch.randn(n_ctx, d, device=device) * 0.02)
        # Save initial copy for deterministic reset (not random each batch)
        self.register_buffer("ctx_init", self.ctx.detach().clone())

        with torch.no_grad():
            token_emb = self.model.token_embedding(self.tokenized.to(device))
        self.register_buffer("token_emb_fixed", token_emb)

        # Insert ctx right after the start token
        self.ctx_pos = 1
        self.register_buffer("attn_mask", (self.tokenized != 0).to(torch.long))

    @torch.no_grad()
    def reset(self):
        """Reset the prompt context to its deterministic initialization."""
        self.ctx.copy_(self.ctx_init)

    def compose_embeds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n_cls, L, d = self.token_emb_fixed.shape
        L_new = L + self.n_ctx
        embeds = torch.zeros((n_cls, L_new, d), device=self.token_emb_fixed.device, dtype=self.token_emb_fixed.dtype)
        mask = torch.zeros((n_cls, L_new), device=self.attn_mask.device, dtype=self.attn_mask.dtype)
        for i in range(n_cls):
            prefix = self.token_emb_fixed[i, : self.ctx_pos, :]
            suffix = self.token_emb_fixed[i, self.ctx_pos :, :]
            embeds[i, : self.ctx_pos, :] = prefix
            embeds[i, self.ctx_pos : self.ctx_pos + self.n_ctx, :] = self.ctx
            embeds[i, self.ctx_pos + self.n_ctx :, :] = suffix

            m_pref = self.attn_mask[i, : self.ctx_pos]
            m_suff = self.attn_mask[i, self.ctx_pos :]
            mask[i, : self.ctx_pos] = m_pref
            mask[i, self.ctx_pos : self.ctx_pos + self.n_ctx] = 1
            mask[i, self.ctx_pos + self.n_ctx :] = m_suff

        # Ensure prompt length equals model context length (e.g., 77)
        max_length = self.model.positional_embedding.shape[0]
        if embeds.shape[1] > max_length:
            embeds = embeds[:, :max_length, :]
            mask = mask[:, :max_length]
        elif embeds.shape[1] < max_length:
            pad_amt = max_length - embeds.shape[1]
            embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad_amt))
            mask = torch.nn.functional.pad(mask, (0, pad_amt))
        return embeds, mask
