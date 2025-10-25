# Copyright (c) 2025.
# Paper-faithful O-TPT implementation (multi-view TPT with view-level confidence selection)
# - Multiple weakly augmented views per test image
# - Keep lowest-entropy views per image
# - Minimize entropy of the average prediction across kept views
# - Add orthogonality regularization on text features
# - Update only prompt context; encoders remain frozen

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from otpt.utils.debug import log, is_debug
except Exception:
    def log(*args, **kwargs):  # fallback no-op
        pass
    def is_debug() -> bool:
        return False


# --------------------------- Common utilities ---------------------------

def _entropy(probs: torch.Tensor) -> torch.Tensor:
    # probs [..., C]
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)


def _orthogonality_loss(text_feats: torch.Tensor) -> torch.Tensor:
    # text_feats: [C, D], rows assumed L2-normalized
    G = text_feats @ text_feats.t()
    I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    return ((G - I) ** 2).mean()


@torch.no_grad()
def infer_logits(model_wrapper, prompt_learner, images: torch.Tensor) -> torch.Tensor:
    """
    Standard CLIP-style inference using the current prompt_learner.
    """
    embeds, mask = prompt_learner.compose_embeds()
    text_feats_all = model_wrapper.encode_text_from_tokens(embeds, mask)  # [C or C*T, D]

    # If multiple templates are supported, mean-pool across them
    if hasattr(prompt_learner, "n_tpl") and getattr(prompt_learner, "n_tpl") and prompt_learner.n_tpl > 1:
        C = getattr(prompt_learner, "num_classes", text_feats_all.shape[0] // prompt_learner.n_tpl)
        T = prompt_learner.n_tpl
        text_feats = text_feats_all.view(C, T, -1).mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    else:
        text_feats = text_feats_all

    img_feats = model_wrapper.encode_image(images)  # [B, D]
    logits = model_wrapper.logit_scale.exp() * (img_feats @ text_feats.t())
    return logits


# --------------------------- TPT-style multi-view generation ---------------------------

def _random_views_tensor(
    images: torch.Tensor,  # [B, 3, H, W], already normalized by preprocess
    views: int,
    min_scale: float,
    max_scale: float,
    hflip_p: float,
    g: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Produce 'views' weak geometric views per image using tensor ops:
      - Random square crop with scale in [min_scale, max_scale], aspect=1
      - Optional horizontal flip
    Returns: [views, B, 3, H, W]
    """
    if views <= 1:
        return images.unsqueeze(0)

    B, C, H, W = images.shape
    device = images.device

    if g is None:
        g = torch.Generator(device=device)
        g.manual_seed(torch.seed())

    out = []
    for _ in range(views):
        # sample per-image crop sizes
        s = (min_scale + (max_scale - min_scale) * torch.rand(B, device=device, generator=g)).clamp(0.5, 1.0)
        batch_crops = []
        for i in range(B):
            side = int(min(H, W) * float(s[i].item()))
            side = max(1, min(side, H, W))
            y_max, x_max = H - side, W - side
            y0 = int(torch.randint(0, y_max + 1, (1,), device=device, generator=g).item()) if y_max > 0 else 0
            x0 = int(torch.randint(0, x_max + 1, (1,), device=device, generator=g).item()) if x_max > 0 else 0
            patch = images[i:i+1, :, y0:y0+side, x0:x0+side]
            patch = F.interpolate(patch, size=(H, W), mode="bilinear", align_corners=False)
            if torch.rand((), device=device, generator=g).item() < hflip_p:
                patch = torch.flip(patch, dims=[-1])
            batch_crops.append(patch)
        out.append(torch.cat(batch_crops, dim=0))  # [B,3,H,W]

    return torch.stack(out, dim=0)  # [views,B,3,H,W]


def _keep_low_entropy_views_per_image(probs_v: torch.Tensor, keep_p: float) -> torch.Tensor:
    """
    probs_v: [V, B, C]
    Returns boolean mask [B, V] that keeps the lowest-entropy fraction per image.
    """
    ent_vb = _entropy(probs_v)              # [V,B]
    ent_bv = ent_vb.transpose(0, 1).contiguous()  # [B,V]
    B, V = ent_bv.shape
    k = max(1, int(round(keep_p * V)))
    idx = ent_bv.argsort(dim=1, descending=False)[:, :k]  # [B,k]
    mask = torch.zeros((B, V), dtype=torch.bool, device=probs_v.device)
    mask.scatter_(1, idx, True)
    return mask  # [B,V]


def _entropy_of_mean_over_kept(probs_v: torch.Tensor, keep_mask_bv: torch.Tensor) -> torch.Tensor:
    """
    probs_v: [V,B,C], keep_mask_bv: [B,V]
    Computes H(mean_v p_v) per image across kept views, then averages over the batch.
    """
    mask = keep_mask_bv.float().unsqueeze(-1)  # [B,V,1]
    mean_p = (probs_v.permute(1, 0, 2) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B,C]
    return _entropy(mean_p).mean()


# --------------------------- O-TPT (paper-faithful) ---------------------------

def otpt_adapt_and_infer(
    model_wrapper,
    prompt_learner,
    images: torch.Tensor,
    tta_steps: int = 1,
    lambda_orth: float = 0.1,
    # Multi-view params (faithful to TPT/O-TPT text)
    tta_views: int = 5,
    view_keep_p: float = 0.6,
    tta_min_scale: float = 0.5,
    tta_max_scale: float = 1.0,
    tta_hflip_p: float = 0.5,
    lr: float = 5e-3,
    # Optional image-level confident selection (disabled by default)
    selection_p: float = -1.0,
) -> torch.Tensor:
    """
    O-TPT with multi-view test-time prompt tuning:
      - For each image in the batch, generate 'tta_views' random views
      - Keep lowest-entropy fraction (view_keep_p) per image
      - Loss = H(mean over kept views) + lambda_orth * L_orth
      - Update only the prompt context via AdamW
      - Encoders (image/text) remain frozen
      - selection_p <= 0 keeps behavior faithful to the paper's view-level selection only
    """
    assert getattr(prompt_learner, "n_ctx", 0) > 0 and prompt_learner.ctx.requires_grad, \
        "O-TPT requires learnable prompt context (n_ctx > 0)."

    amp_ctx = nullcontext()
    optim = torch.optim.AdamW([prompt_learner.ctx], lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    for step in range(tta_steps):
        with amp_ctx, torch.enable_grad():
            # Compose text features (with grad through prompt context)
            embeds, mask = prompt_learner.compose_embeds()
            text_feats_all = model_wrapper.encode_text_from_tokens(embeds, mask)  # [C or C*T, D]

            if hasattr(prompt_learner, "n_tpl") and getattr(prompt_learner, "n_tpl") and prompt_learner.n_tpl > 1:
                C = getattr(prompt_learner, "num_classes", text_feats_all.shape[0] // prompt_learner.n_tpl)
                T = prompt_learner.n_tpl
                text_feats = text_feats_all.view(C, T, -1).mean(dim=1)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            else:
                text_feats = text_feats_all

            # Create multiple views per image (no grad through image branch)
            with torch.no_grad():
                views = _random_views_tensor(
                    images,
                    views=tta_views,
                    min_scale=tta_min_scale,
                    max_scale=tta_max_scale,
                    hflip_p=tta_hflip_p,
                )  # [V,B,3,H,W]
                img_feats_v = torch.stack(
                    [model_wrapper.encode_image(views[v]) for v in range(views.size(0))],
                    dim=0,
                )  # [V,B,D]

            # View-wise class probabilities
            sims_v = torch.einsum("vbd,cd->vbc", img_feats_v, text_feats)  # [V,B,C]
            probs_v = torch.softmax(sims_v, dim=-1)                         # [V,B,C]

            # View-level confidence selection: keep low-entropy views per image
            keep_mask_bv = _keep_low_entropy_views_per_image(probs_v, keep_p=view_keep_p)  # [B,V]

            # Entropy of average prob across kept views (marginal entropy)
            loss_ent = _entropy_of_mean_over_kept(probs_v, keep_mask_bv)

            # Optional: image-level confident selection (disabled by default)
            if selection_p is not None and selection_p > 0:
                # Compute marginal probs per image and focus loss on most confident images
                mask = keep_mask_bv.float().unsqueeze(-1)  # [B,V,1]
                mean_p = (probs_v.permute(1, 0, 2) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)  # [B,C]
                ent_img = _entropy(mean_p)  # [B]
                k = max(1, int(round(ent_img.numel() * selection_p)))
                idx = ent_img.topk(k, largest=False).indices  # lowest-entropy images
                # Scale loss to reflect fraction selected
                loss_ent = loss_ent * (idx.numel() / ent_img.numel())

            # Orthogonality regularization on normalized text features
            loss_orth = _orthogonality_loss(text_feats)

            loss = loss_ent + lambda_orth * loss_orth

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([prompt_learner.ctx], max_norm=1.0)
        optim.step()

        if is_debug():
            log(f"[O-TPT][step {step+1}/{tta_steps}] H(mean)={float(loss_ent):.4f}, "
                f"L_orth={float(loss_orth):.4f}, views={tta_views}, keep_p={view_keep_p}")

    # Final logits on the original (single) view for reporting
    with torch.no_grad():
        return infer_logits(model_wrapper, prompt_learner, images)


# --------------------------- Temperature scaling helpers ---------------------------

@torch.no_grad()
def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    return logits / max(T, 1e-6)


@torch.no_grad()
def compute_ece(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> torch.Tensor:
    confidences, preds = probs.max(dim=1)
    accuracies = preds.eq(labels)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device, dtype=probs.dtype)
    ece = probs.new_tensor(0.0)
    for i in range(num_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            ece = ece + (mask.float().mean()) * (bin_conf - bin_acc).abs()
    return ece


@torch.no_grad()
def nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -log_probs[torch.arange(labels.numel(), device=labels.device), labels].mean()


@torch.no_grad()
def find_best_temperature(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    metric: str = "ece",
    num_bins: int = 15,
    T_min: float = 0.5,
    T_max: float = 5.0,
    steps: int = 40,
) -> float:
    best_T, best_val = 1.0, float("inf")
    grid = torch.linspace(T_min, T_max, steps=steps)
    for T in grid:
        scaled = apply_temperature(logits, float(T))
        if metric == "entropy" or labels is None:
            log_probs = torch.log_softmax(scaled, dim=1)
            probs = log_probs.exp()
            val = (-(probs * log_probs).sum(dim=1)).mean().item()
        elif metric == "nll":
            val = nll_from_logits(scaled, labels).item()
        else:
            probs = torch.softmax(scaled, dim=1)
            val = compute_ece(probs, labels, num_bins=num_bins).item()
        if val < best_val:
            best_val, best_T = val, float(T)
    if is_debug():
        log(f"[TS] best_T={best_T:.3f}, metric={metric}, value={best_val:.6f}")
    return best_T
