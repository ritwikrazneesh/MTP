import argparse
import torch
from tqdm import tqdm

from otpt.utils.debug import set_debug, log, is_debug
from otpt.data.rs_imagefolder import build_imagefolder_eval
from otpt.models.remoteclip_adapter import build_remoteclip_via_openclip as build_remoteclip, RemoteCLIPWrapper as RCLIP_Wrap, PromptLearner as RCLIP_Prompt
from otpt.models.openclip_adapter import build_openclip, OpenClipWrapper as OCLIP_Wrap, PromptLearner as OCLIP_Prompt
from otpt.models.otpt_core import infer_logits, otpt_adapt_and_infer, find_best_temperature, apply_temperature

def parse_args():
    p = argparse.ArgumentParser("O-TPT / RemoteCLIP RS eval with canonical class names + temperature scaling")
    # Data
    p.add_argument("--dataset", type=str, required=True, choices=[
        "eurosat", "patternnet", "nwpu-resisc45", "ucm", "whu-rs19", "aid"
    ])
    p.add_argument("--data-root", type=str, default=".")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--canonicalize-classes", action="store_true", help="Map folder class names to canonical phrases for prompts")

    # Backend / Model
    p.add_argument("--backend", type=str, default="remoteclip", choices=["remoteclip", "openclip"])
    p.add_argument("--model-name", type=str, default="ViT-B-32")
    p.add_argument("--pretrained-id", type=str, default="laion2b_s34b_b88k")  # for openclip fallback
    p.add_argument("--pretrained-ckpt", type=str, default="", help="Path to RemoteCLIP .pt checkpoint (required if backend=remoteclip)")

    # Prompt / O-TPT
    p.add_argument("--mode", type=str, default="zeroshot", choices=["zeroshot", "otpt"])
    p.add_argument("--n-ctx", type=int, default=0, help="Context tokens for prompts (use 0 for strict zeroshot)")
    p.add_argument("--template", type=str, default="a satellite photo of a {}.")
    p.add_argument("--tta-steps", type=int, default=1)
    p.add_argument("--selection-p", type=float, default=0.1)
    p.add_argument("--lambda-orth", type=float, default=0.1)
    p.add_argument("--otpt-lr", type=float, default=5e-3)

    # Temperature scaling
    p.add_argument("--temp-scale", action="store_true", help="Enable post-hoc temperature scaling before metrics")
    p.add_argument("--temp-metric", type=str, default="ece", choices=["ece", "nll", "entropy"], help="Metric to tune T")
    p.add_argument("--temp-range", type=float, nargs=2, default=[0.5, 5.0], help="Range [min max] for T grid search")
    p.add_argument("--temp-steps", type=int, default=40, help="Grid steps for T search")
    p.add_argument("--ece-bins", type=int, default=15, help="Bins for ECE when temp-metric=ece")

    # Debug
    p.add_argument("--debug", action="store_true", help="Verbose logging")

    args = p.parse_args()
    set_debug(args.debug)
    return args

def main():
    args = parse_args()

        # ... inside main() right after args parsing ...
    if args.mode == "otpt" and args.n_ctx <= 0:
        raise SystemExit("O-TPT requires --n-ctx > 0 (e.g., --n-ctx 8 or 16).")

    
    device = args.device if torch.cuda.is_available() else "cpu"

    # Build model + preprocess + tokenizer
    if args.backend == "remoteclip":
        if not args.pretrained_ckpt:
            raise SystemExit("--backend remoteclip requires --pretrained-ckpt path to RemoteCLIP checkpoint (.pt).")
        model_raw, tokenizer, preprocess = build_remoteclip(
            model_name=args.model_name,
            checkpoint_path=args.pretrained_ckpt,
            device=device,
        )
        modelw = RCLIP_Wrap(model_raw).to(device)
        Prompt = RCLIP_Prompt
    else:
        model_raw, tokenizer, preprocess = build_openclip(
            model_name=args.model_name,
            pretrained=args.pretrained_id,
            device=device,
        )
        modelw = OCLIP_Wrap(model_raw).to(device)
        Prompt = OCLIP_Prompt

    loaders, classnames = build_imagefolder_eval(
        dataset_name=args.dataset,
        data_root=args.data_root,
        preprocess=preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_canonical=args.canonicalize_classes,
    )
    _, val_loader = loaders

    # Prompt learner with canonicalized names in the dataset's label order
    pl = Prompt(model_raw, tokenizer, classnames, n_ctx=args.n_ctx, template=args.template, device=device).to(device)

    if is_debug():
        log(f"[RUN] dataset={args.dataset}, backend={args.backend}, model={args.model_name}, mode={args.mode}")
        log(f"[RUN] classes={len(classnames)}; first 5={classnames[:5]}")

    modelw.eval()
    pl.eval()

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Eval", dynamic_ncols=True):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if args.mode == "otpt" and args.n_ctx > 0:
                logits = otpt_adapt_and_infer(
                    model_wrapper=modelw,
                    prompt_learner=pl,
                    images=images,
                    tta_steps=args.tta_steps,
                    lambda_orth=args.lambda_orth,
                    selection_p=args.selection_p,
                    lr=args.otpt_lr,
                )
                # reset prompt for next batch adaptation
                pl.reset()
            else:
                logits = infer_logits(modelw, pl, images)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # Argmax predictions (same before/after TS)
    preds = logits.argmax(dim=1)
    probs = torch.softmax(logits, dim=1)

    top1 = (preds == labels).float().mean().item()

    # Optional Temperature Scaling
    if args.temp_scale:
        T = find_best_temperature(
            logits,
            labels=None if args.temp_metric == "entropy" else labels,
            metric=args.temp_metric,
            num_bins=args.ece_bins,
            T_min=min(args.temp_range),
            T_max=max(args.temp_range),
            steps=args.temp_steps,
        )
        logits_scaled = apply_temperature(logits, T)
        probs_scaled = torch.softmax(logits_scaled, dim=1)
        # ECE after TS
        ece = _compute_ece(probs_scaled, labels, num_bins=args.ece_bins)
        print(f"Top-1: {top1*100:.2f}%, ECE (TS): {ece*100:.2f}%  [T*={T:.3f}]")
    else:
        ece = _compute_ece(probs, labels, num_bins=args.ece_bins)
        print(f"Top-1: {top1*100:.2f}%, ECE: {ece*100:.2f}%")

def _compute_ece(probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
    confidences, preds = probs.max(dim=1)
    accuracies = preds.eq(labels)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    N = probs.size(0)
    for i in range(num_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.any():
            bin_acc = accuracies[mask].float().mean().item()
            bin_conf = confidences[mask].mean().item()
            ece += (mask.float().mean().item()) * abs(bin_conf - bin_acc)
    return ece

if __name__ == "__main__":
    main()
