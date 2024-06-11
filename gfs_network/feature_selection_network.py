import torch
import torch.nn as nn

def gumbel_sigmoid(logits, tau: float = 1, hard: bool = False, threshold: float = 0.5):
    gumbels = (
        -torch.empty_like(logits).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels.sigmoid()

    if hard:
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

class FeatureSelectionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, architecture=None):
        super(FeatureSelectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc1.weight.data.zero_()

        if architecture is not None:
            self.cont = architecture
        else:
            self.cont = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

    def forward(self, orig, temperature=1):
        x = self.fc1(orig)
        mask = gumbel_sigmoid(x, tau=temperature, hard=temperature == 0)
        
        x = orig * mask
        x = self.cont(x)
        return x, mask.mean()