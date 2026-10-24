from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class ElmanNN(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 10, output_size: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity='tanh',
            batch_first=True,
            bias=True,
        )
        self.w_output = nn.Linear(hidden_size, output_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)
        nn.init.zeros_(self.rnn.bias_hh_l0)
        nn.init.xavier_uniform_(self.w_output.weight)
        nn.init.zeros_(self.w_output.bias)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq shape: [batch, steps, input_size]
        out, _ = self.rnn(x_seq)
        return self.w_output(out[:, -1, :])


@dataclass
class ENNTrainResult:
    model: ElmanNN
    train_losses: List[float]
    test_nmse: float


def _nmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    return torch.mean((y_pred - y_true) ** 2)


def train_enn_single_run(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden_size: int,
    epochs: int = 100,
    lr: float = 0.01,
    device: str = 'cuda',
) -> ENNTrainResult:
    dev = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    model = ElmanNN(input_size=6, hidden_size=hidden_size, output_size=1).to(dev)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Stateful Elman recurrence: process windows as one ordered sequence per epoch.
    xtr = torch.tensor(x_train[None, :, :], dtype=torch.float32, device=dev)  # [1, N, 6]
    ytr = torch.tensor(y_train[:, None], dtype=torch.float32, device=dev)
    xte = torch.tensor(x_test[None, :, :], dtype=torch.float32, device=dev)  # [1, Ntest, 6]
    yte = torch.tensor(y_test[:, None], dtype=torch.float32, device=dev)

    losses: List[float] = []
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        h0 = torch.zeros(1, 1, hidden_size, dtype=torch.float32, device=dev)
        out, _ = model.rnn(xtr, h0)  # [1, N, H]
        pred = model.w_output(out.squeeze(0))  # [N, 1]
        loss = _nmse(pred, ytr)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        h0 = torch.zeros(1, 1, hidden_size, dtype=torch.float32, device=dev)
        out, _ = model.rnn(xte, h0)  # [1, Ntest, H]
        pred = model.w_output(out.squeeze(0))  # [Ntest, 1]
        test_nmse = float(_nmse(pred, yte).cpu())

    return ENNTrainResult(model=model, train_losses=losses, test_nmse=test_nmse)
