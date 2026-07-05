"""Vectorized multi-seed training for AutoNFS.

`adaptive_balance_search` needs `n_seeds` independent trainings per grid
point. Training them one by one wastes almost all the time on per-op
dispatch overhead (the network is tiny). Instead, this module stacks the
parameters of `n_members` independently initialized networks along a leading
dimension and trains them all in one loop with batched matmuls.

The sum of the per-member losses has a block-diagonal dependency structure,
so its gradient w.r.t. each member's parameters equals that member's own
loss gradient, and Adam updates are elementwise -- training the stacked
ensemble is mathematically identical to `n_members` sequential runs (up to
RNG draw order).
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from typing import Literal

from .feature_selection_network import FeatureSelectionNetwork
from .select_gumbel_features import _make_adam, _single_thread_ctx


def train_gumbel_ensemble(
    X,
    y,
    n_members: int,
    device="cpu",
    verbose=False,
    temperature_decay=0.997,
    epochs=150,
    batch_size=32,
    fs_balance=1.0,
    mode: Literal["classification", "regression"] = "classification",
):
    """Train `n_members` FeatureSelectionNetworks at once and return their
    feature vote counts as an int32 array of shape (n_members, n_features).

    `X` must already be scaled and `y` one-hot encoded (classification) or
    shaped (N, 1) (regression), exactly as `AutoNFS.fit` prepares them.
    Uses `target_features_mode="raw"` semantics (the mode `AutoNFS.fit`
    passes): loss = criterion + fs_balance * mean(mask).
    """
    X = X.to(device=device, dtype=torch.float32)
    y = y.to(device=device, dtype=torch.float32)
    n_features = X.shape[1]

    if mode == "classification":
        class_weight = (y.sum(axis=0) / len(y)).flip(0)
        n_out = len(class_weight)
        target = y.argmax(dim=1)
    elif mode == "regression":
        n_out = 1
        target = y
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Initialize members exactly like sequential runs would, then stack.
    members = [FeatureSelectionNetwork(n_features, 32, n_out) for _ in range(n_members)]

    def stacked(get, lr_group):
        p = torch.stack([get(m).detach() for m in members]).to(device).requires_grad_()
        lr_group.append(p)
        return p

    fc1_params, cont_params = [], []
    emb = torch.stack([m.emb.detach() for m in members]).to(device)  # (M, 1, 32); frozen, like in select_gumbel_features
    W1 = stacked(lambda m: m.fc1.weight, fc1_params)                 # (M, F, 32)
    b1 = stacked(lambda m: m.fc1.bias, fc1_params)                   # (M, F)
    cont = [
        (stacked(lambda m, i=i: m.cont[i].weight, cont_params),
         stacked(lambda m, i=i: m.cont[i].bias, cont_params))
        for i in (0, 2, 4)
    ]

    optimizer = _make_adam([
        {"params": fc1_params, "lr": 4e-3},
        {"params": cont_params, "lr": 3e-4},
    ])

    X_batches = [X[i : i + batch_size] for i in range(0, len(X), batch_size)]
    y_batches = [target[i : i + batch_size] for i in range(0, len(target), batch_size)]
    if mode == "classification":
        # Per-batch normalizers of the weighted cross-entropy mean, so the
        # flattened reduction="sum" below equals the sum of per-member
        # weighted-mean CE losses.
        weight_sums = [class_weight[y_].sum() for y_ in y_batches]
    else:
        weight_sums = [None] * len(y_batches)

    temperature = 2.0
    with _single_thread_ctx(device):
        for epoch in tqdm.trange(epochs, disable=not verbose):
            for X_, y_, wsum in zip(X_batches, y_batches, weight_sums):
                optimizer.zero_grad(set_to_none=True)

                logits = torch.baddbmm(b1.unsqueeze(1), emb, W1.transpose(1, 2))  # (M, 1, F)
                gumbels = -torch.empty_like(logits).exponential_().log()
                mask = torch.sigmoid((logits + gumbels) / temperature)

                x = X_.unsqueeze(0) * mask                                        # (M, B, F)
                for i, (W, b) in enumerate(cont):
                    x = torch.baddbmm(b.unsqueeze(1), x, W.transpose(1, 2))
                    if i < len(cont) - 1:
                        x = torch.relu(x)

                selected = mask.mean(dim=(1, 2))                                  # (M,)
                if mode == "classification":
                    ce = F.cross_entropy(
                        x.reshape(-1, n_out), y_.repeat(n_members),
                        weight=class_weight, reduction="sum",
                    )
                    loss = ce / wsum + fs_balance * selected.sum()
                else:
                    loss = ((x - y_) ** 2).mean(dim=(1, 2)).sum() + fs_balance * selected.sum()

                loss.backward()
                optimizer.step()
            temperature = temperature * temperature_decay

        # Same stochastic tau=0 vote as select_gumbel_features, per member.
        with torch.no_grad():
            logits = torch.baddbmm(b1.unsqueeze(1), emb, W1.transpose(1, 2))      # (M, 1, F)
            gumbels = -torch.empty(
                n_members, len(X_batches), n_features, device=logits.device
            ).exponential_().log()
            votes = ((logits + gumbels) > 0).sum(dim=1).to(torch.int32)

    return votes.cpu().numpy()
