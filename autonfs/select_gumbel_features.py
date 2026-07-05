from .feature_selection_network import FeatureSelectionNetwork
import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from typing import Literal


def _make_adam(param_groups):
    """Adam with the fused kernel when the device supports it (single fused
    call per step instead of one op per parameter tensor)."""
    try:
        return optim.Adam(param_groups, fused=True)
    except (RuntimeError, ValueError):
        return optim.Adam(param_groups)


@contextlib.contextmanager
def _single_thread_ctx(device):
    """Run CPU training single-threaded: the per-step tensors are so small
    that intra-op thread synchronization costs more than it saves."""
    is_cpu = (device == "cpu") or (getattr(device, "type", None) == "cpu")
    if not is_cpu:
        yield
        return
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev)


def select_gumbel_features(
    X_train,
    y_train,
    device="cpu",
    verbose=False,
    temperature_decay=0.997,
    epochs=300,
    batch_size=1,
    fs_balance=1.0,
    target_features_mode: Literal["auto", "target", "raw"] = "auto",
    mode: Literal["classification", "regression"] = "classification",
):
    X = X_train.to(device=device, dtype=torch.float32)
    y = y_train.to(device=device, dtype=torch.float32)

    if mode == "classification":
        balance = (y.sum(axis=0) / len(y)).flip(0)
        network = FeatureSelectionNetwork(X.shape[1], 32, len(balance)).to(device)
        criterion = nn.CrossEntropyLoss(weight=balance)
        target = y.argmax(dim=1)
    elif mode == "regression":
        network = FeatureSelectionNetwork(X.shape[1], 32, 1).to(device)
        criterion = nn.MSELoss()
        target = y
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Slice the data into batches once, up front; iterating a DataLoader
    # re-collates every batch on every epoch, which dominates the runtime
    # for a network this small.
    X_batches = [X[i : i + batch_size] for i in range(0, len(X), batch_size)]
    y_batches = [target[i : i + batch_size] for i in range(0, len(target), batch_size)]

    optimizer = _make_adam(
        [
            {"params": network.fc1.parameters(), "lr": 4e-3},
            {"params": network.cont.parameters(), "lr": 3e-4},
        ]
    )
    temperature = 2.0

    if target_features_mode == "auto":
        if X.shape[1] > 10000:
            target_features_mode = "raw"
        else:
            target_features_mode = "target"

    if target_features_mode == "target":
        target_features = 2
        target_features = target_features / X.shape[1]

    elif target_features_mode == "raw":
        target_features = None


    with _single_thread_ctx(device):
        for epoch in tqdm.trange(epochs, disable=not verbose):
            for X_, y_ in zip(X_batches, y_batches):
                optimizer.zero_grad(set_to_none=True)
                output, selected_no = network(X_, temperature=temperature)

                if target_features is not None:
                    loss = criterion(output, y_) + fs_balance * (selected_no - target_features) ** 2
                else:
                    loss = criterion(output, y_) + fs_balance * selected_no
                loss.backward()
                optimizer.step()
            temperature = temperature * temperature_decay

        # Stochastic vote over hard (tau=0) gumbel-sigmoid masks: one vote per
        # training batch, all drawn in a single vectorized sample. A tau=0 mask
        # selects feature j iff logits_j + gumbel_noise > 0.
        with torch.no_grad():
            logits = network.fc1(network.emb)
            n_votes = len(X_batches)
            gumbels = -torch.empty(n_votes, logits.shape[1], device=logits.device).exponential_().log()
            features_sum = ((logits + gumbels) > 0).sum(dim=0).to(torch.int32)

    return features_sum.cpu().numpy(), network
