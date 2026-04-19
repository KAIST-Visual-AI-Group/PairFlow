import math
import typing

import torch
import torchmetrics

LOG2 = math.log(2)


class NLL(torchmetrics.aggregation.MeanMetric):
  def update(self,
             value:typing.Union[float, torch.Tensor],
             weight:typing.Union[float, torch.Tensor]=1.0) -> None:
    """Update state with data.

    Args:
      value: Either a float or tensor containing data.
        Additional tensor dimensions will be flattened
      weight: Either a float or tensor containing weights
        for calculating the average. Shape of weight should
        be able to broadcast with the shape of `value`.
        Default to `1.0` corresponding to simple harmonic
        average.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=self.dtype,
                                device=self.device)
    if (weight is not None
        and not isinstance(weight, torch.Tensor)):
      weight = torch.as_tensor(weight,
                               dtype=self.dtype,
                               device=self.device)
    weight = torch.broadcast_to(weight, value.shape)
    value, weight = self._cast_and_nan_check_input(value,
                                                   weight)

    if value.numel() == 0:
      return
    self.mean_value += value.sum()
    self.weight += weight.sum()


class BPD(NLL):
  def compute(self) -> torch.Tensor:
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
  def compute(self) -> torch.Tensor:
    return torch.exp(self.mean_value / self.weight)


class Metrics:
  def __init__(self, **kwargs) -> None:
    del kwargs
    metrics = torchmetrics.MetricCollection({
        'nll': NLL(), 'bpd': BPD(), 'ppl': Perplexity()})
    metrics.set_dtype(torch.float64)
    self.train_nlls = metrics.clone(prefix='train/')
    self.train_aux = BPD()
    self.valid_nlls = metrics.clone(prefix='val/')
    self.valid_aux = BPD()

  def to(self, *args, **kwargs):
    self.train_nlls = self.train_nlls.to(*args, **kwargs)
    self.train_aux = self.train_aux.to(*args, **kwargs)
    self.valid_nlls = self.valid_nlls.to(*args, **kwargs)
    self.valid_aux = self.valid_aux.to(*args, **kwargs)

  def reset(self):
    self.train_nlls.reset()
    self.train_aux.reset()
    self.valid_nlls.reset()
    self.valid_aux.reset()

  def update_train(self, nll, aux_loss, num_tokens):
    self.train_nlls.update(nll, num_tokens)
    self.train_aux.update(aux_loss, num_tokens)

  def update_valid(self, nll, aux_loss, num_tokens):
    self.valid_nlls.update(nll, num_tokens)
    self.valid_aux.update(aux_loss, num_tokens)
