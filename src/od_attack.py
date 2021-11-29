from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from util import non_max_suppression


class ODPGDAttackBase(ABC):

  def __init__(self,
               conf_thres: float = 0.15,
               iou_thres: float = 0.9,
               alpha: float = 0.0005,
               eps: float = 0.005,
               max_iter: int = 20):
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


class YOLOv5VanishAttack(ODPGDAttackBase, nn.Module):

  def __init__(self,
               yolov5: nn.Module,
               conf_thres: float = 0.15,
               iou_thres: float = 0.9,
               alpha: float = 0.0005,
               eps: float = 0.005,
               max_iter: int = 20):
    super().__init__(conf_thres=conf_thres, iou_thres=iou_thres,
                     alpha=alpha, eps=eps, max_iter=max_iter)

    if 'AutoShape' in str(type(yolov5)):
      yolov5 = yolov5.model

    for m in yolov5.modules():
      if hasattr(m, 'inplace'):
        m.inplace = False

    self.__yolov5 = yolov5

  def forward(self, x: torch.Tensor, **kargs: Dict[str, Any]) -> torch.Tensor:
    conf_thres = self._conf_thres if 'conf_thres' not in kargs else kargs['conf_thres']
    iou_thres = self._iou_thres if 'iou_thres' not in kargs else kargs['iou_thres']
    alpha = self._alpha if 'alpha' not in kargs else kargs['alpha']
    eps = self._eps if 'eps' not in kargs else kargs['eps']
    max_iter = self._max_iter if 'max_iter' not in kargs else kargs['max_iter']
    verbose = False if 'verbose' not in kargs else kargs['verbose']

    if verbose:
      pbar = tqdm(total=max_iter, leave=True, desc='Generating AE ...')

    x = x.to(next(self.__yolov5.parameters()))
    x_adv = x
    for _ in range(max_iter):
      x_adv.detach_().requires_grad_(True)

      y_pred = non_max_suppression(self.__yolov5(x_adv)[0], conf_thres, iou_thres)[0]
      if y_pred.size(0) == 0:  # none of objects are detected.
        break

      conf = y_pred[:, 4]
      loss = F.binary_cross_entropy(conf, conf.new_zeros(size=conf.shape), reduction='mean')
      self.__yolov5.zero_grad()
      loss.backward()

      x_adv = x_adv.detach_() - alpha * x_adv.grad.sign()
      x_adv = (x + (x_adv - x).clip(-eps, eps)).clip(0, 1)

      if verbose:
        pbar.update(1)

    if verbose:
      steps = pbar.n
      pbar.update(pbar.total - pbar.n)
      pbar.set_description(f'Done (in {steps} steps)')
      pbar.close()

    return x_adv
