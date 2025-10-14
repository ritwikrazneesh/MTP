import argparse
import numpy as np
import torch
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import balanced_accuracy_score, log_loss
from tqdm import tqdm

from otpt.data.registry import get_dataset
from otpt.eval.metrics import expected_calibration_error
from otpt.models.otpt_core import infer_logits, otpt_adapt_and_infer
from otpt.models.openclip_adapter import build_openclip, OpenClipWrapper as OCLIP_Wrap, PromptLearner as OCLIP_Prompt
from otpt.models.remoteclip_adapter import build_remoteclip_via_openclip, RemoteCLIPWrapper as RCLIP_Wrap, PromptLearner as RCLIP_Prompt


def evaluate(loader, modelw, pl, mode: str, tta_steps: int, lambda_orth: float, selection_p: float, device: str):
    modelw.eval()
    acc_metric = MulticlassAccuracy(num_classes=len(pl.classnames)).to(device)
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc=f"Eval ({mode})"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mode == "zeroshot":
            logits = infer_logits(modelw, pl, images)
        elif mode == "otpt":
            with torch.no_grad():
                pl.ctx.normal_(mean=0.0, std=0.02)  # reset per batch
            logits = otpt_adapt_and_infer(
                modelw, pl, images,
                tta_steps=tta_steps,
                lambda_orth=lambda_orth,
                selection_p=selection_p
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        acc_metric.update(preds, labels)
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    top1 = acc_metric.compute().item()
    probs_np = np.concatenate(all_probs, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    bal_acc = balanced_accuracy_score(labels_np, probs_np.argmax(axis=1))
    eps = 1e-12
    nll = log_loss(labels_np, np.clip(probs_np, eps, 1 - eps), labels=list(range(probs_np.shape[1])))
    ece = expected_calibration_error(probs_np, labels_np, n_bins=15)
    return {"top1": top1, "balanced_acc": bal_acc, "nll": nll, "ece": ece*100}


def main():
    p = argparse.ArgumentParser("O-TPT single-entry CLI (RemoteCLIP/OpenCLIP)")
    # Core
    p.add_argument("--dataset", type=str, default="eurosat")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--mode", type=str, default="zeroshot", choices=["zeroshot", "otpt"])
    p.add_argument("--backend", type=str, default="remoteclip", choices=["remoteclip", "openclip"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda")
    # Prompting / templates
    p.add_argument("--template", type=str, default="a satellite photo of a {}.")
    p.add_argument("--n-ctx", type=int, default=8)
    # O-TPT hyperparams
    p.add_argument("--tta-steps", type=int, default=1)
    p.add_argument("--lambda-orth", type=float, default=0.1)
    p.add_argument("--selection-p", type=float, default=0.1)
    # OpenCLIP identifiers
    p.add_argument("--model-name", type=str, default="ViT-B-32")
    p.add_argument("--pretrained-id", type=str, default="laion2b_s34b_b88k")  # OpenCLIP fallback
    # RemoteCLIP checkpoint path (per README using open-clip)
    p.add_argument("--pretrained-ckpt", type=str, default="", help="Path to RemoteCLIP .pt checkpoint (e.g., remoteclip_vitb32.pt)")

    args = p.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # Build model + preprocess + tokenizer
    if args.backend == "remoteclip":
        if not args.pretrained_ckpt:
            raise SystemExit("--backend remoteclip requires --pretrained-ckpt path to RemoteCLIP checkpoint (.pt).")
        model_raw, tokenizer, preprocess = build_remoteclip_via_openclip(
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

    # Dataset
    loaders, classnames = get_dataset(
        name=args.dataset,
        data_root=args.data_root,
        preprocess=preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    _, val_loader = loaders

    # Prompt learner
    pl = Prompt(model_raw, tokenizer, classnames, n_ctx=args.n_ctx, template=args.template, device=device).to(device)

    # Eval
    metrics = evaluate(
        val_loader, modelw, pl,
        mode=args.mode,
        tta_steps=args.tta_steps,
        lambda_orth=args.lambda_orth,
        selection_p=args.selection_p,
        device=device,
    )
    print(f"[{args.backend}][{args.dataset}][{args.mode}] -> {metrics}")


if __name__ == "__main__":
    main()
