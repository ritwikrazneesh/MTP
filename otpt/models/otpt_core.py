import torch
from contextlib import nullcontext
from torch.cuda.amp import autocast as cuda_autocast, GradScaler as CudaGradScaler


def avg_entropy_softmax(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1).clamp_min(1e-8)
    return -(probs * probs.log()).sum(dim=1).mean()


def select_confident_indices(logits: torch.Tensor, percentile: float = 0.1) -> torch.Tensor:
    probs = logits.softmax(dim=1).clamp_min(1e-8)
    ent = -(probs * probs.log()).sum(dim=1)
    k = max(1, int(len(ent) * percentile))
    return ent.topk(k, largest=False).indices


def orthogonality_loss(ctx: torch.Tensor) -> torch.Tensor:
    G = ctx @ ctx.t()  # (n_ctx, n_ctx)
    I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    return ((G - I) ** 2).mean()


@torch.no_grad()
def infer_logits(model_wrapper, prompt_learner, images: torch.Tensor) -> torch.Tensor:
    embeds, mask = prompt_learner.compose_embeds()
    text_feats = model_wrapper.encode_text_from_tokens(embeds, mask)
    img_feats = model_wrapper.encode_image(images)
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
    device = prompt_learner.ctx.device
    use_cuda_amp = (device.type == "cuda")
    scaler = CudaGradScaler(enabled=use_cuda_amp)
    amp_ctx = cuda_autocast() if use_cuda_amp else nullcontext()

    optim = torch.optim.AdamW([prompt_learner.ctx], lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    for _ in range(tta_steps):
        with amp_ctx:
            embeds, mask = prompt_learner.compose_embeds()
            text_feats = model_wrapper.encode_text_from_tokens(embeds, mask)
            img_feats = model_wrapper.encode_image(images)
            logits = model_wrapper.logit_scale.exp() * (img_feats @ text_feats.t())

            idx = select_confident_indices(logits.detach(), percentile=selection_p)
            loss_ent = avg_entropy_softmax(logits[idx])
            loss_orth = orthogonality_loss(prompt_learner.ctx)
            loss = loss_ent + lambda_orth * loss_orth

        optim.zero_grad(set_to_none=True)
        if use_cuda_amp:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()

    with torch.no_grad():
        embeds, mask = prompt_learner.compose_embeds()
        text_feats = model_wrapper.encode_text_from_tokens(embeds, mask)
        img_feats = model_wrapper.encode_image(images)
        logits = model_wrapper.logit_scale.exp() * (img_feats @ text_feats.t())
    return logits
