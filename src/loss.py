import abc
from abc import ABC

import torch
import torch.nn as nn


class AbstractLoss(ABC):
    @abc.abstractmethod
    def compute(self, *args, **kwargs):
        pass


class Loss(AbstractLoss):
    def __init__(self, criterion: nn.Module, model: nn.Module, optimizer, scheduler, max_grad_norm):
        self._criterion = criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm

    def compute(self, logits: torch.tensor, targets: torch.tensor, mode: str):
        """

        :param logits: shape (batch_size, tgt_seq_length - 1, vocab_size)
        :param targets: shape (batch_size, tgt_seq_length - 1)
        :param mode:
        :return:
        """
        logits = logits.view(logits.size(0) * logits.size(1), -1)
        targets = targets.contiguous().view(-1)
        loss = self._criterion(logits, targets)

        if mode == 'train':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()
            self._model.zero_grad()

        return loss.item()
