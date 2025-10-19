from typing import List, Tuple
import torch
import torch.nn as nn
import open_clip

# A compact prompt bank for RS zero-shot; your passed template will be added on top
RS_TEMPLATES = [
    "a satellite photo of a {}.",
    "an aerial view of a {}.",
    "a remote sensing image of a {}.",
    "an overhead photo of a {}.",
    "overhead satellite imagery of a {}.",
    "a high-resolution aerial image of a {}.",
    "a photo of a {} from above.",
]

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
        text = token_embeddings + self.model.positional_embedding.to(token_embeddings.dtype)
        context_length = text.shape[1]
        causal_mask = torch.full((context_length, context_length), float("-inf"), device=text.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        text = self.model.transformer(text, attn_mask=causal_mask)
        text = self.model.ln_final(text)
        lengths = attn_mask.sum(dim=1) - 1  # Use attn_mask only to find EOT
        x = text[torch.arange(text.shape[0], device=text.device), lengths]
        x = x @ self.model.text_projection
        return x / x.norm(dim=-1, keepdim=True)


class PromptLearner(nn.Module):
    def __init__(self, model, tokenizer, classnames: List[str], n_ctx: int = 8, template: str = "a satellite photo of a {}.", device: str = "cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classnames = [c.replace("_", " ").replace("-", " ").strip() for c in classnames]
        self.num_classes = len(self.classnames)
        self.n_ctx = n_ctx

        # Build template bank: user template (first) + RS defaults (dedup)
        tpl_list = [template] + RS_TEMPLATES
        # Deduplicate while preserving order
        seen = set()
        self.templates = [t for t in tpl_list if not (t in seen or seen.add(t))]
        self.n_tpl = len(self.templates)

        # Flatten prompts: per-class, per-template
        prompts = [tpl.format(name) for name in self.classnames for tpl in self.templates]
        tokenized = self.tokenizer(prompts)
        self.register_buffer("tokenized", tokenized)

        d = self.model.transformer.width
        self.ctx = nn.Parameter(torch.randn(n_ctx, d, device=device) * 0.02)
        # Deterministic reset state (optional for O-TPT)
        self.register_buffer("ctx_init", self.ctx.detach().clone())

        with torch.no_grad():
            token_emb = self.model.token_embedding(self.tokenized.to(device))
        self.register_buffer("token_emb_fixed", token_emb)

        # Insert ctx at position 1 (after [SOS])
        self.ctx_pos = 1
        self.register_buffer("attn_mask", (self.tokenized != 0).to(torch.long))

    @torch.no_grad()
    def reset(self):
        if hasattr(self, "ctx_init"):
            self.ctx.copy_(self.ctx_init)

    def compose_embeds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return token embeddings (with inserted ctx) and attention mask for all class-template prompts.
        Shapes:
          token_emb_fixed: [C*T, L, d]
          returns embeds: [C*T, L+n_ctx, d], mask: [C*T, L+n_ctx]
        """
        N, L, d = self.token_emb_fixed.shape  # N == C * T
        L_new = L + self.n_ctx
        device = self.token_emb_fixed.device
        dtype = self.token_emb_fixed.dtype

        embeds = torch.zeros((N, L_new, d), device=device, dtype=dtype)
        mask = torch.zeros((N, L_new), device=device, dtype=self.attn_mask.dtype)

        # Fill per-prompt
        # NOTE: Looping over N keeps memory bounded and is acceptable at eval time
        for i in range(N):
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
