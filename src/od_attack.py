from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .util import query_dict, non_max_suppression


class ODPGDAttackBase(ABC):

  def __init__(self, conf_thres: float, iou_thres: float, alpha: float, eps: float, max_iter: float):
    super().__init__()

    self._conf_thres = conf_thres
    self._iou_thres = iou_thres
    self._alpha = alpha
    self._eps = eps
    self._max_iter = max_iter

  @abstractmethod
  def forward(self, *args, **kargs) -> Any:
    pass

  @property
  def conf_thres(self) -> float:
    return self._conf_thres

  @property
  def iou_thres(self) -> float:
    return self._iou_thres

  @property
  def alpha(self) -> float:
    return self._alpha

  @property
  def eps(self) -> float:
    return self._eps

  @property
  def max_iter(self) -> int:
    return self._max_iter


class YOLOv5PGDAttackBase(ODPGDAttackBase, nn.Module):

  def __init__(self, model: nn.Module, *args: Tuple[Any, ...], **kargs: Dict[str, Any]):
    super().__init__(*args, **kargs)

    if 'AutoShape' in str(type(model)):
      model = model.model

    for m in model.modules():  # disable all inplace operations to calculate loss.
      if hasattr(m, 'inplace'):
        m.inplace = False

    self._model = model

  @property
  def yolov5(self) -> nn.Module:
    return self._model


class YOLOv5VanishAttack(YOLOv5PGDAttackBase):

  def forward(self, x: Union[np.ndarray, torch.Tensor], **kargs: Dict[str, Any]) -> torch.Tensor:
    conf_thres = query_dict(kargs, 'conf_thres', self._conf_thres)
    iou_thres = query_dict(kargs, 'iou_thres', self._iou_thres)
    alpha = query_dict(kargs, 'alpha', self._alpha)
    eps = query_dict(kargs, 'eps', self._eps)
    max_iter = query_dict(kargs, 'max_iter', self._max_iter)
    verbose = query_dict(kargs, 'verbose', False)

    if isinstance(x, np.ndarray):
      x = torch.from_numpy(x)
      x = x.permute(2, 0, 1) if x.size(-1) == 3 else x

    input_shape = x.shape
    if x.ndim != 3:
      if (x.ndim == 4) and (1 < x.size(0)):
        raise ValueError('Batched input to YOLOv5VanishAttack is unsupported.')
      elif (x.ndim < 4) or (4 < x.ndim):
        raise ValueError('The passed input tensor is invalid shape.')
    else:
       x = x.unsqueeze(0)

    x = x.to(next(self._model.parameters())).detach()

    if verbose:
      pbar = tqdm(total=max_iter, leave=True, desc='Generating AE ...')

    x_adv = x.clone()
    for _ in range(max_iter):
      x_adv.requires_grad_(True)

      y_pred = non_max_suppression(self._model(x_adv)[0], conf_thres, iou_thres)[0]
      if y_pred.size(0) == 0:  # none of objects are detected.
        break

      conf = y_pred[:, 4]
      loss = F.binary_cross_entropy(conf, conf.new_zeros(size=conf.shape), reduction='mean')
      self._model.zero_grad()
      loss.backward()

      x_adv.detach_()
      x_adv = x_adv - alpha * x_adv.grad.sign()
      x_adv = (x + (x_adv - x).clip(-eps, eps)).clip(0, 1)

      if verbose:
        pbar.update(1)

    if verbose:
      steps = pbar.n
      pbar.update(pbar.total - pbar.n)
      pbar.set_description(f'Done (in {steps} steps)')
      pbar.close()

    return x_adv.view(input_shape)
