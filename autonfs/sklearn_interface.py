from .select_gumbel_features import select_gumbel_features
from .adaptive import adaptive_balance_search, DEFAULT_GRID
import torch
from typing import Literal, Union


class AutoNFS:
    def __init__(
        self,
        batch_size=32,
        temperature_decay: float = 0.997,
        epochs: int = 150,
        balance: Union[float, Literal["auto"]] = 1.0,
        device: str = "cpu",
        verbose: bool = False,
        mode: Literal["classification", "regression"] = "classification",
        adaptive_grid=DEFAULT_GRID,
        adaptive_n_seeds: int = 5,
        adaptive_threshold_frac: float = 0.9,
        adaptive_val_size: float = 0.25,
        random_state: int = 0,
    ) -> None:
        """Perform feature selection using GFSNetwork.

        Args:
            batch_size (int, optional): Batch size. Larger batch size speeds up training, but makes feature selection less aggresive. Defaults to 32 (batch_size=1 was found to cause severe GPU/CPU slowdowns with no feature-selection benefit over 32/64; see HPO study).
            temperature_decay (float, optional): Temperature decay. Defaults to 0.997.
            epochs (int, optional): Number of epochs. Defaults to 150 (150-300 are statistically equivalent; 150 is fastest).
            balance (float or "auto", optional): Balance between classification and feature selection. Larger value puts more weight on feature selection (fewer, more predictive features per feature), but risks collapsing the mask entirely (0 features selected) on some datasets. Defaults to 1.0 (maximum selection pressure; WARNING: this was found to cause complete feature-mask collapse in the majority of seeds on several benchmark datasets -- see HPO study). Pass `balance="auto"` to adaptively back off from 1.0 on a held-out split, one log-grid step at a time, stopping as soon as the mask no longer collapses and downstream score stays within `adaptive_threshold_frac` of an all-features baseline -- this is the recommended setting for a dataset you have not already tuned `balance` for. See `autonfs.adaptive.adaptive_balance_search` and the HPO study addendum for details; the chosen value and full backoff trajectory are stored on `self.adaptive_result_`.
            device (str, optional): Device to use. Defaults to "cpu".
            verbose (bool, optional): Verbosity. Defaults to False.
            adaptive_grid (tuple, optional): Log-spaced grid of balance values to try, highest (most aggressive) first, when `balance="auto"`. Defaults to (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001).
            adaptive_n_seeds (int, optional): Seeds evaluated per grid point when `balance="auto"`; the backoff decision is based on the median across these seeds. Defaults to 5.
            adaptive_threshold_frac (float, optional): Minimum fraction of the all-features baseline's downstream score that a grid point's median score must retain to be accepted, when `balance="auto"`. Defaults to 0.9.
            adaptive_val_size (float, optional): Held-out fraction of the training data used to score each grid point during the `balance="auto"` search. Defaults to 0.25.
            random_state (int, optional): Random state for the train/validation split used by `balance="auto"`. Defaults to 0.
        """
        self.scores_ = None
        self.device = device
        self.verbose = verbose
        self.temperature_decay = temperature_decay
        self.epochs = epochs
        self.network = None
        self.batch_size = batch_size
        self.balance = balance
        self.mode = mode
        self.adaptive_grid = adaptive_grid
        self.adaptive_n_seeds = adaptive_n_seeds
        self.adaptive_threshold_frac = adaptive_threshold_frac
        self.adaptive_val_size = adaptive_val_size
        self.random_state = random_state
        self.adaptive_result_ = None

    def fit(
        self,
        X,
        y,
        scale=True,
        target_features_mode: Literal["auto", "target", "raw"] = "raw",
    ):
        balance = self.balance
        if balance == "auto":
            balance = self._resolve_adaptive_balance(X, y)

        # cast to torch
        if type(X) is not torch.Tensor:
            X = torch.tensor(X)
            y = torch.tensor(y)

        # assert that y is one-hot encoded
        if len(y.shape) == 1 and self.mode == "classification":
            y = torch.nn.functional.one_hot(y.to(int))
        elif len(y.shape) == 1 and self.mode == "regression":
            y = y.view(-1, 1).to(torch.float32)
            
        if scale:
            # perform whitening
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)

        self.scores_, self.network = select_gumbel_features(
            X,
            y,
            self.device,
            self.verbose,
            temperature_decay=self.temperature_decay,
            epochs=self.epochs,
            batch_size=self.batch_size,
            fs_balance=balance,
            target_features_mode=target_features_mode,
            mode=self.mode,
        )
        self.balance_ = balance
        self.ranking_ = self.scores_.argsort()[::-1]
        return self

    def _resolve_adaptive_balance(self, X, y):
        """Run the balance="auto" backoff search on a held-out split of (X, y)
        and return the chosen balance. Stores the full result (chosen value,
        reason, and per-grid-point trajectory) on `self.adaptive_result_`.
        """
        import numpy as np
        from sklearn.model_selection import train_test_split

        X_np = X.numpy() if hasattr(X, "numpy") else np.asarray(X)
        y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
        stratify = y_np if self.mode == "classification" else None
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_np, y_np, test_size=self.adaptive_val_size,
            random_state=self.random_state, stratify=stratify,
        )
        self.adaptive_result_ = adaptive_balance_search(
            X_tr, y_tr, X_val, y_val,
            mode=self.mode,
            grid=self.adaptive_grid,
            n_seeds=self.adaptive_n_seeds,
            threshold_frac=self.adaptive_threshold_frac,
            batch_size=self.batch_size,
            epochs=self.epochs,
            temperature_decay=self.temperature_decay,
            device=self.device,
            verbose=self.verbose,
        )
        if self.verbose:
            print(f"[AutoNFS] balance='auto' -> chosen balance="
                  f"{self.adaptive_result_.chosen_balance} "
                  f"({self.adaptive_result_.chosen_reason})")
        return self.adaptive_result_.chosen_balance

    @property
    def support_(self):
        assert self.scores_ is not None, "You must call fit before accessing support_"
        return self.scores_ > 0

    def transform(self, X):
        assert self.support_ is not None, "You must call fit before transform"
        return X[:, self.support_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
