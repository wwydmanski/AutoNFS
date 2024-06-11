from .select_gumbel_features import select_gumbel_features
import torch


class GFSNetwork:
    def __init__(
        self,
        threshold: float = 0.01,
        device: str = "cpu",
        verbose: bool = False,
        temperature_decay: float = 0.9999,
        epochs: int = 300,
    ) -> None:
        """Perform feature selection using GFSNetwork.

        Args:
            threshold (float, optional): Keep features with scores above this percentile score. Defaults to 0.01.
            device (str, optional): Device to use. Defaults to "cpu".
            verbose (bool, optional): Verbosity. Defaults to False.
            temperature_decay (float, optional): Temperature decay. Defaults to 0.9999.
            epochs (int, optional): Number of epochs. Defaults to 300.
        """
        self.threshold = threshold
        self.scores_ = None
        self.device = device
        self.verbose = verbose
        self.temperature_decay = temperature_decay
        self.epochs = epochs

    def fit(self, X, y):
        # cast to torch
        if type(X) != torch.Tensor:
            X = torch.tensor(X)
            y = torch.tensor(y)

        # assert that y is one-hot encoded
        if len(y.shape) == 1:
            y = torch.nn.functional.one_hot(y.to(int))

        self.scores_ = select_gumbel_features(
            X,
            y,
            self.device,
            self.verbose,
            temperature_decay=self.temperature_decay,
            epochs=self.epochs,
        )
        self.ranking_ = self.scores_.argsort()[::-1]
        return self

    @property
    def support_(self):
        assert self.scores_ is not None, "You must call fit before accessing support_"
        perc_score = self.scores_ / self.scores_.max()
        return self.scores_ > perc_score

    def transform(self, X):
        assert self.support_ is not None, "You must call fit before transform"
        return X[:, self.support_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
