from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from util import non_max_suppression


class YOLOv5VanishAttack(nn.Module):

  def __init__(self,
               yolov5: nn.Module,
               conf_thres: float = 0.15,
               iou_thres: float = 0.9,
               alpha: float = 0.0005,
               eps: float = 0.005,
               max_iter: int = 20):
    super().__init__()

    if 'AutoShape' in str(type(yolov5)):
      yolov5 = yolov5.model

    for m in yolov5.modules():
      if hasattr(m, 'inplace'):
        m.inplace = False

    self.__yolov5 = yolov5
    self.__conf_thres = conf_thres
    self.__iou_thres = iou_thres
    self.__alpha = alpha
    self.__eps = eps
    self.__max_iter = max_iter

  def forward(self, x: torch.Tensor, **kargs: Dict[str, Any]) -> torch.Tensor:
    conf_thres = self.__conf_thres if 'conf_thres' not in kargs else kargs['conf_thres']
    iou_thres = self.__iou_thres if 'iou_thres' not in kargs else kargs['iou_thres']
    alpha = self.__alpha if 'alpha' not in kargs else kargs['alpha']
    eps = self.__eps if 'eps' not in kargs else kargs['eps']
    max_iter = self.__max_iter if 'max_iter' not in kargs else kargs['max_iter']
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

  @property
  def conf_thres(self) -> float:
    return self.__conf_thres

  @property
  def iou_thres(self) -> float:
    return self.__iou_thres

  @property
  def alpha(self) -> float:
    return self.__alpha

  @property
  def eps(self) -> float:
    return self.__eps

  @property
  def max_iter(self) -> int:
    return self.__max_iter
