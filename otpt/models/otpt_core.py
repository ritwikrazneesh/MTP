import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler as CudaGradScaler

def avg_entropy_softmax_stable(logits: torch.Tensor) -> torch.Tensor:
    """
    Numerically-stable entropy:
      H(p) = -sum_i p_i * log p_i, with p = softmax(logits)
    Use log_softmax to avoid overflow/underflow.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=1).mean()

def select_confident_indices_stable(logits: torch.Tensor, percentile: float = 0.1) -> torch.Tensor:
    """
    Select lowest-entropy samples (most confident) using a stable entropy estimator.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=1)
    k = max(1, int(ent.numel() * percentile))
    return ent.topk(k, largest=False).indices

def orthogonality_loss_on_text(text_feats: torch.Tensor) -> torch.Tensor:
    """
    Encourage orthogonality among class text features.
    text_feats: [C, D], assumed L2-normalized per row.
    Penalize (W W^T - I).
    """
    G = text_feats @ text_feats.t()               # [C, C], cosine similarities since rows are normalized
    I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    return ((G - I) ** 2).mean()

@torch.no_grad()
def infer_logits(model_wrapper, prompt_learner, images: torch.Tensor) -> torch.Tensor:
    embeds, mask = prompt_learner.compose_embeds()
    text_feats = model_wrapper.encode_text_from_tokens(embeds, mask)  # [C, D], normalized in wrapper
    img_feats = model_wrapper.encode_image(images)                    # [B, D], normalized in wrapper
    return model_wrapper.logit_scale.exp() * (img_feats @ text_feats.t())

def otpt_adapt_and_infer(
    model_wrapper,
    prompt_learner,
    images: torch.Tensor,
    tta_steps: int = 1,
    lambda_orth: float = 0.1,
    selection_p: float = 0.1,
    lr: float = 5e-3,
) -> torch.Tensor:
    """
    Test-time prompt adaptation loop with stable losses and fp32 to prevent NaN grads.
    Key differences vs before:
      - Use log_softmax-based entropy and confidence selection (stable).
      - Compute loss on cosine similarities (no logit_scale in loss path).
      - Detach image features (only adapt text/prompt).
      - Apply orthogonality on TEXT FEATURES (not raw ctx).
      - Disable AMP here; clip grads; guard non-finite steps.
    """
    # Force fp32 in adaptation to avoid fp16 instability on small prompt tensors
    use_cuda_amp = False
    scaler = CudaGradScaler(enabled=False)
    amp_ctx = nullcontext()

    optim = torch.optim.AdamW([prompt_learner.ctx], lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    for _ in range(tta_steps):
        with amp_ctx:
            # Compose text features for current prompt
            embeds, mask = prompt_learner.compose_embeds()
            text_feats = model_wrapper.encode_text_from_tokens(embeds, mask)  # [C, D], normalized

            # Image features are fixed (not adapted); detach graph
            with torch.no_grad():
                img_feats = model_wrapper.encode_image(images)                # [B, D], normalized

            # Use cosine similarities for adaptation (no exp(logit_scale) here)
            sims = img_feats @ text_feats.t()                                 # [B, C]

            # Select confident subset and compute stable entropy
            idx = select_confident_indices_stable(sims.detach(), percentile=selection_p)
            loss_ent = avg_entropy_softmax_stable(sims[idx])

            # Orthogonality on text features
            loss_orth = orthogonality_loss_on_text(text_feats)

            loss = loss_ent + lambda_orth * loss_orth

        optim.zero_grad(set_to_none=True)
        loss.backward()

        # Clip and guard against non-finite grads (prevents stuck NaN steps)
        grad_norm = torch.nn.utils.clip_grad_norm_([prompt_learner.ctx], max_norm=1.0)
        if not torch.isfinite(grad_norm):
            optim.zero_grad(set_to_none=True)
            continue

        optim.step()

    # Final inference (with logit_scale) for metrics/reporting
    with torch.no_grad():
        embeds, mask = prompt_learner.compose_embeds()
        text_feats = model_wrapper.encode_text_from_tokens(embeds, mask)
        img_feats = model_wrapper.encode_image(images)
        logits = model_wrapper.logit_scale.exp() * (img_feats @ text_feats.t())
    return logits
