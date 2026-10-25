"""Report test accuracy from cached features and released checkpoints."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--datasets", nargs="+", choices=("waterbirds", "celeba", "multinli", "civilcomments"),
                        default=["waterbirds", "celeba", "multinli", "civilcomments"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(f"{'Dataset':14s} {'Model':5s} {'mAcc':>7s} {'WGA':>7s}")
    for dataset in args.datasets:
        data = torch.load(args.cache_dir / f"{dataset}.pt", map_location="cpu", weights_only=True)
        features, labels, groups = data["f"].float(), data["y"].flatten(), data["p"].flatten()
        pairs = torch.stack((labels, groups), 1).unique(dim=0)
        for method in ("erm", "nsf"):
            state = torch.load(args.checkpoint_dir / f"{dataset}_{method}.pt", map_location="cpu", weights_only=True)
            x = features
            if "transformation.w" in state:
                x = (x + state["transformation.b"]) * state["transformation.w"] - state["transformation.b"]
            logits = F.linear(x, state["fc.weight"], state["fc.bias"])
            pred = (logits[:, 0] > 0).long() if logits.shape[1] == 1 else logits.argmax(1)
            macc = (pred == labels).float().mean().item()
            wga = min((pred[(labels == y) & (groups == p)] == y).float().mean().item() for y, p in pairs)
            print(f"{dataset:14s} {method.upper():5s} {100*macc:7.2f} {100*wga:7.2f}")


if __name__ == "__main__":
    main()
