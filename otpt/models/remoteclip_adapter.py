from typing import List, Tuple
import torch
import torch.nn as nn
import open_clip
from otpt.utils.debug import log, is_debug

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
        feats = feats / feats.norm(dim=-1, keepdim=True)
        if is_debug():
            log(f"[ENC] image feats shape={tuple(feats.shape)}")
        return feats

    def encode_text_from_tokens(self, token_embeddings: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        text = token_embeddings + self.model.positional_embedding.to(token_embeddings.dtype)
        context_length = text.shape[1]
        causal_mask = torch.full((context_length, context_length), float("-inf"), device=text.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        text = self.model.transformer(text, attn_mask=causal_mask)
        text = self.model.ln_final(text)
        lengths = attn_mask.sum(dim=1) - 1
        x = text[torch.arange(text.shape[0], device=text.device), lengths]
        x = x @ self.model.text_projection
        x = x / x.norm(dim=-1, keepdim=True)
        if is_debug():
            log(f"[ENC] text feats shape={tuple(x.shape)}")
        return x

class PromptLearner(nn.Module):
    def __init__(self, model, tokenizer, classnames: List[str], n_ctx: int = 8, template: str = "a satellite photo of a {}.", device: str = "cuda"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.classnames = [c.replace("_", " ").replace("-", " ").strip() for c in classnames]
        self.num_classes = len(self.classnames)
        self.n_ctx = n_ctx

        tpl_list = [template] + RS_TEMPLATES
        seen, templates = set(), []
        for t in tpl_list:
            if t not in seen:
                templates.append(t); seen.add(t)
        self.templates = templates
        self.n_tpl = len(self.templates)

        prompts = [tpl.format(name) for name in self.classnames for tpl in self.templates]
        tokenized = self.tokenizer(prompts)
        self.register_buffer("tokenized", tokenized)

        d = self.model.transformer.width
        self.ctx = nn.Parameter(torch.randn(n_ctx, d, device=device) * 0.02)
        self.register_buffer("ctx_init", self.ctx.detach().clone())

        with torch.no_grad():
            token_emb = self.model.token_embedding(self.tokenized.to(device))
        self.register_buffer("token_emb_fixed", token_emb)

        self.ctx_pos = 1
        self.register_buffer("attn_mask", (self.tokenized != 0).to(torch.long))

        if is_debug():
            log(f"[PL] classes={self.num_classes}, templates={self.n_tpl}, n_ctx={self.n_ctx}, token_emb_fixed={tuple(self.token_emb_fixed.shape)}")

    @torch.no_grad()
    def reset(self):
        self.ctx.copy_(self.ctx_init)
        if is_debug():
            log("[PL] reset ctx to deterministic init")

    def compose_embeds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable assembly: embeds = cat(prefix, ctx, suffix).
        """
        N, L, d = self.token_emb_fixed.shape
        device = self.token_emb_fixed.device

        prefix = self.token_emb_fixed[:, : self.ctx_pos, :]        # [N, 1, d]
        suffix = self.token_emb_fixed[:, self.ctx_pos :, :]        # [N, L-1, d]

        if self.n_ctx > 0:
            ctx_exp = self.ctx.unsqueeze(0).expand(N, -1, -1)     # [N, n_ctx, d], requires_grad=True
            embeds = torch.cat([prefix, ctx_exp, suffix], dim=1)  # [N, L+n_ctx, d]
            ones_ctx = torch.ones((N, self.n_ctx), device=device, dtype=self.attn_mask.dtype)
            mask = torch.cat([self.attn_mask[:, : self.ctx_pos], ones_ctx, self.attn_mask[:, self.ctx_pos :]], dim=1)
        else:
            embeds = torch.cat([prefix, suffix], dim=1)
            mask = torch.cat([self.attn_mask[:, : self.ctx_pos], self.attn_mask[:, self.ctx_pos :]], dim=1)

        max_length = self.model.positional_embedding.shape[0]
        if embeds.shape[1] > max_length:
            embeds = embeds[:, :max_length, :]
            mask = mask[:, :max_length]
        elif embeds.shape[1] < max_length:
            pad_amt = max_length - embeds.shape[1]
            embeds = torch.nn.functional.pad(embeds, (0, 0, 0, pad_amt))
            mask = torch.nn.functional.pad(mask, (0, pad_amt))

        if is_debug():
            log(f"[PL] compose_embeds -> embeds={tuple(embeds.shape)}, mask={tuple(mask.shape)}")
        return embeds, mask
