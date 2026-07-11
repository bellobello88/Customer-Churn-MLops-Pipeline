import torch
from torch import nn


class ChurnModel(nn.Module):
    """
    Multilayer perceptron for binary customer churn prediction.

    Input:
        Numeric customer features

    Output:
        One raw logit for each customer
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.network(features)

        # Convert shape from [batch_size, 1] to [batch_size].
        return logits.squeeze(1)