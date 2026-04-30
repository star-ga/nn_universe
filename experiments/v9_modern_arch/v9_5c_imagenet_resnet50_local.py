"""V9.5c — ImageNet-1K ResNet-50 90-epoch from-scratch trajectory (LOCAL RTX 3080).

Local-friendly: AMP fp16 mixed precision, batch 256, lr 0.1, cosine, momentum 0.9, wd 1e-4.
RTX 3080 (10 GB) + i7-5930K. Wall time est ~25-35 hours.
FIM measured at epochs 0/10/30/60/90 with 200-probe float64 protocol.
"""
import argparse, json, time, os, io, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from torchvision.models import resnet50
from datasets import load_dataset
from PIL import Image


def gini(v):
    v = np.asarray(v, dtype=np.float64).flatten(); v = v[v >= 0]
    if v.size == 0 or v.sum() == 0: return 0.0
    v.sort(); n = v.size; cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def effective_rank(v):
    v = np.asarray(v, dtype=np.float64).flatten(); v = v[v > 0]
    if v.size == 0: return 0.0
    return float((v.sum() ** 2) / (v.size * (v ** 2).sum()))


def top_1pct_mass(v):
    v = np.asarray(v, dtype=np.float64).flatten(); v = v[v > 0]
    if v.size == 0: return 0.0
    s = np.sort(v)[::-1]; k1 = max(1, int(s.size * 0.01))
    return float(s[:k1].sum() / s.sum())


def tier_ratio(v):
    s = np.sort(v)[::-1]; n = len(s)
    k1 = max(1, int(n * 0.01)); k3 = max(1, int(n * 0.5))
    t1 = float(s[:k1].mean()); t3 = float(s[-k3:].mean())
    if t3 <= 0:
        nz = s[s > 0]
        t3 = float(nz[-max(len(nz)//10, 1):].mean()) if len(nz) else 1e-30
    return t1 / t3 if t3 > 0 else float("inf")


class HFImageNet(Dataset):
    def __init__(self, split, transform, hf_path):
        os.environ.setdefault('HF_HOME', '/data/checkpoints/hf_cache')
        self.ds = load_dataset(hf_path, split=split, num_proc=4, keep_in_memory=False)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        rec = self.ds[idx]
        img = rec["image"]
        if isinstance(img, dict) and "bytes" in img:
            img = Image.open(io.BytesIO(img["bytes"]))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return self.transform(img), int(rec["label"])


def imagenet_loaders(hf_path, batch=256, workers=8):
    tt = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tv = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    train = HFImageNet("train", tt, hf_path)
    val = HFImageNet("val", tv, hf_path)
    print(f"train={len(train)} val={len(val)}", flush=True)
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True),
        DataLoader(val, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True),
    )


def train_epoch(net, loader, opt, scaler, device, ep, total_eps):
    net.train()
    correct = 0; total = 0; loss_sum = 0.0
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with autocast(dtype=torch.float16):
            out = net(x); loss = F.cross_entropy(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
        if i % 100 == 0:
            print(f"  ep{ep:>2}/{total_eps} step {i}/{len(loader)}  loss={loss.item():.3f}  acc={correct/max(total,1):.3f}  elapsed={time.time()-t0:.0f}s", flush=True)
    return loss_sum / total, correct / total, time.time() - t0


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(dtype=torch.float16):
            out = net(x)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


def fim_diagonal(net, loader, device, n_probes):
    fim = {n: torch.zeros_like(p, dtype=torch.float64) for n, p in net.named_parameters()}
    net.eval()
    iter_loader = iter(loader)
    seen = 0
    while seen < n_probes:
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader); x, y = next(iter_loader)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        for i in range(min(x.size(0), n_probes - seen)):
            xi, yi = x[i:i+1], y[i:i+1]
            out = net(xi); loss = F.cross_entropy(out, yi)
            net.zero_grad(set_to_none=True); loss.backward()
            for n, p in net.named_parameters():
                if p.grad is not None:
                    fim[n] += p.grad.data.double() ** 2
            seen += 1
    for n in fim: fim[n] /= n_probes
    return torch.cat([v.flatten() for v in fim.values()]).cpu().numpy()


def measure(net, val_loader, device, n_probes):
    fim = fim_diagonal(net, val_loader, device, n_probes)
    ratio = tier_ratio(fim)
    return {"T1T3": ratio, "log10_T1T3": float(np.log10(max(ratio, 1e-30))),
            "gini": gini(fim), "eff_rank_n": effective_rank(fim) / fim.size,
            "top_1pct_mass": top_1pct_mass(fim)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-path", default="/data/checkpoints/imagenet_1k_hf")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--checkpoint-epochs", type=int, nargs="+", default=[0, 10, 30, 60, 90])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--n-probes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "v9_5c_imagenet_resnet50_fromscratch_results.json"))
    args = ap.parse_args()

    os.environ.setdefault('HF_HOME', '/data/checkpoints/hf_cache')

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    print(f"=== ImageNet-1K ResNet-50 from-scratch (LOCAL, AMP fp16) === seed={args.seed} epochs={args.epochs} batch={args.batch}", flush=True)
    train_loader, val_loader = imagenet_loaders(args.hf_path, batch=args.batch, workers=args.workers)

    net = resnet50(weights=None).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"ResNet-50 params: {n_params:,}", flush=True)
    print(f"GPU free after load: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB", flush=True)

    opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler()

    trajectory = []
    if 0 in args.checkpoint_epochs:
        print(f"\n[ep 0 / random init] FIM measurement", flush=True)
        t0 = time.time()
        m = measure(net, val_loader, device, args.n_probes)
        m["epoch"] = 0; m["test_accuracy"] = float(evaluate(net, val_loader, device))
        m["fim_time_s"] = time.time() - t0
        trajectory.append(m)
        print(f"  T1/T3={m['T1T3']:.3e}  Gini={m['gini']:.4f}  acc={m['test_accuracy']:.3f}  ({m['fim_time_s']:.1f}s)", flush=True)
        partial = {"config": vars(args), "n_params": int(n_params), "trajectory": trajectory, "partial": True}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(partial, indent=2, default=str))

    for ep in range(args.epochs):
        train_loss, train_acc, dt = train_epoch(net, train_loader, opt, scaler, device, ep+1, args.epochs)
        sched.step()
        ep_idx = ep + 1
        if ep_idx in args.checkpoint_epochs:
            print(f"\n[ep {ep_idx}/{args.epochs}] train_acc={train_acc:.3f} ({dt:.0f}s); FIM...", flush=True)
            t1 = time.time()
            m = measure(net, val_loader, device, args.n_probes)
            m["epoch"] = ep_idx
            m["test_accuracy"] = float(evaluate(net, val_loader, device))
            m["train_accuracy"] = float(train_acc); m["train_loss"] = float(train_loss)
            m["fim_time_s"] = time.time() - t1; m["epoch_time_s"] = dt
            trajectory.append(m)
            print(f"  T1/T3={m['T1T3']:.3e}  Gini={m['gini']:.4f}  test_acc={m['test_accuracy']:.3f}", flush=True)
            partial = {"config": vars(args), "n_params": int(n_params), "trajectory": trajectory, "partial": True}
            Path(args.out).write_text(json.dumps(partial, indent=2, default=str))

    log10 = [(m["epoch"], m["log10_T1T3"]) for m in trajectory]
    drop = log10[0][1] - log10[-1][1] if len(log10) >= 2 else 0
    payload = {
        "config": vars(args), "n_params": int(n_params), "trajectory": trajectory,
        "log10_T1T3_init_vs_final": {
            "epoch_0": log10[0][1] if log10 else None,
            "epoch_final": log10[-1][1] if log10 else None,
            "log10_drop_during_training": drop,
            "T1T3_drop_factor": 10 ** drop if log10 else None,
        },
        "interpretation": "Full ImageNet-1K ResNet-50 90-epoch from-scratch with FIM at 0/10/30/60/90 (LOCAL RTX 3080 + AMP fp16).",
    }
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n=== Final ===")
    for m in trajectory:
        print(f"  ep={m['epoch']:>2} log10(T1/T3)={m['log10_T1T3']:.3f} Gini={m['gini']:.3f} test_acc={m['test_accuracy']:.3f}")
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
