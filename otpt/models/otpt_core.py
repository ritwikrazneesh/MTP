import torch
from contextlib import nullcontext
from torch.cuda.amp import GradScaler as CudaGradScaler
from otpt.utils.debug import log, is_debug

def avg_entropy_softmax_stable(logits: torch.Tensor) -> torch.Tensor:
    # Stable H(p) with p=softmax(logits)
    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    h = -(probs * log_probs).sum(dim=1).mean()
    if is_debug():
        log(f"[O-TPT] avg_entropy: {h.item():.6f}")
    return h

def select_confident_indices_stable(logits: torch.Tensor, percentile: float = 0.1) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    ent = -(probs * log_probs).sum(dim=1)  # [B]
    k = max(1, int(ent.numel() * percentile))
    idx = ent.topk(k, largest=False).indices
    if is_debug():
        log(f"[O-TPT] selection_p={percentile}, select_k={k}, ent[min]={ent.min().item():.6f}, ent[max]={ent.max().item():.6f}")
    return idx

def orthogonality_loss_on_text(text_feats: torch.Tensor) -> torch.Tensor:
    # text_feats: [C, D], rows L2-normalized
    G = text_feats @ text_feats.t()  # [C, C]
    I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    loss = ((G - I) ** 2).mean()
    if is_debug():
        g_diag = torch.diag(G)
        offdiag = (G - torch.diag(g_diag)).abs().mean().item()
        log(f"[O-TPT] ortho: mean_offdiag={offdiag:.6f}, loss={loss.item():.6f}")
    return loss

@torch.no_grad()
def infer_logits(model_wrapper, prompt_learner, images: torch.Tensor) -> torch.Tensor:
    embeds, mask = prompt_learner.compose_embeds()
    text_feats_all = model_wrapper.encode_text_from_tokens(embeds, mask)
    if hasattr(prompt_learner, "n_tpl") and prompt_learner.n_tpl > 1:
        C = getattr(prompt_learner, "num_classes", text_feats_all.shape[0] // prompt_learner.n_tpl)
        T = prompt_learner.n_tpl
        text_feats = text_feats_all.view(C, T, -1).mean(dim=1)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        if is_debug():
            log(f"[ZS] templates={T}, classes={C}, text_feats_all={tuple(text_feats_all.shape)}, text_feats={tuple(text_feats.shape)}")
    else:
        text_feats = text_feats_all
        if is_debug():
            log(f"[ZS] templates=1, text_feats={tuple(text_feats.shape)}")

    img_feats = model_wrapper.encode_image(images)
    if is_debug():
        log(f"[ZS] img_feats={tuple(img_feats.shape)}, text_feats={tuple(text_feats.shape)}")
    logits = model_wrapper.logit_scale.exp() * (img_feats @ text_feats.t())
    if is_debug():
        finite_ratio = torch.isfinite(logits).float().mean().item()
        log(f"[ZS] logits finite ratio={finite_ratio:.3f}, logit_scale={model_wrapper.logit_scale.exp().item():.3f}")
    return logits

def otpt_adapt_and_infer(
    model_wrapper,
    prompt_learner,
    images: torch.Tensor,
    tta_steps: int = 1,
    lambda_orth: float = 0.1,
    selection_p: float = 0.1,
    lr: float = 5e-3,
) -> torch.Tensor:
    use_cuda_amp = False  # stability for small prompt params
    scaler = CudaGradScaler(enabled=False)
    amp_ctx = nullcontext()

    optim = torch.optim.AdamW([prompt_learner.ctx], lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    for step in range(tta_steps):
        with amp_ctx:
            embeds, mask = prompt_learner.compose_embeds()
            text_feats_all = model_wrapper.encode_text_from_tokens(embeds, mask)  # [C*T, D]
            if hasattr(prompt_learner, "n_tpl") and prompt_learner.n_tpl > 1:
                C = getattr(prompt_learner, "num_classes", text_feats_all.shape[0] // prompt_learner.n_tpl)
                T = prompt_learner.n_tpl
                text_feats = text_feats_all.view(C, T, -1).mean(dim=1)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            else:
                text_feats = text_feats_all

            with torch.no_grad():
                img_feats = model_wrapper.encode_image(images)  # [B, D]

            sims = img_feats @ text_feats.t()  # [B, C], cosine similarities (no exp(logit_scale))
            idx = select_confident_indices_stable(sims.detach(), percentile=selection_p)
            loss_ent = avg_entropy_softmax_stable(sims[idx])
            loss_orth = orthogonality_loss_on_text(text_feats)
            loss = loss_ent + lambda_orth * loss_orth

        if is_debug():
            log(f"[O-TPT][{step+1}/{tta_steps}] ctx_norm={prompt_learner.ctx.norm().item():.6f}, loss_ent={loss_ent.item():.6f}, loss_orth={loss_orth.item():.6f}, total={loss.item():.6f}")

        optim.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_([prompt_learner.ctx], max_norm=1.0)
        if not torch.isfinite(grad_norm):
            if is_debug():
                log(f"[O-TPT][{step+1}] non-finite grad_norm={grad_norm}, skipping step")
            optim.zero_grad(set_to_none=True)
            continue

        optim.step()

        if is_debug():
            log(f"[O-TPT][{step+1}] ctx_norm_after={prompt_learner.ctx.norm().item():.6f}")

    with torch.no_grad():
        embeds, mask = prompt_learner.compose_embeds()
        text_feats_all = model_wrapper.encode_text_from_tokens(embeds, mask)
        if hasattr(prompt_learner, "n_tpl") and prompt_learner.n_tpl > 1:
            C = getattr(prompt_learner, "num_classes", text_feats_all.shape[0] // prompt_learner.n_tpl)
            T = prompt_learner.n_tpl
            text_feats = text_feats_all.view(C, T, -1).mean(dim=1)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        else:
            text_feats = text_feats_all

        img_feats = model_wrapper.encode_image(images)
        logits = model_wrapper.logit_scale.exp() * (img_feats @ text_feats.t())
        if is_debug():
            log(f"[O-TPT] final logits: finite_ratio={torch.isfinite(logits).float().mean().item():.3f}, logit_scale={model_wrapper.logit_scale.exp().item():.3f}")
    return logits
