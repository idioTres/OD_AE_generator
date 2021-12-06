import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision


def query_dict(kargs: Dict[str, Any], key: str, default_value: Optional[Any] = None) -> Any:
  return kargs[key] if key in kargs else default_value


def load_image(img_path: str) -> Optional[np.ndarray]:
  if not os.path.exists(img_path):
    return None

  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img


def mksquare(img: np.ndarray,
             pad_color: Tuple[float, float, float] = (114, 114, 114),
             size: Optional[Tuple[int, int]] = None) -> np.ndarray:
  h, w, *_ = img.shape
  vpad, hpad = [max(0, w - h) // 2] * 2, [max(0, h - w) // 2] * 2
  img = cv2.copyMakeBorder(img, *vpad, *hpad, cv2.BORDER_CONSTANT, value=pad_color)
  img = cv2.resize(img, size) if size is not None else img
  return img


def xywh2xyxy(xywh: torch.Tensor) -> torch.Tensor:
  xyxy = xywh.clone().detach_()
  xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
  xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
  xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
  xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
  return xyxy


# https://github.com/ultralytics/yolov5/blob/master/utils/general.py
def non_max_suppression(y_pred: torch.Tensor,
                        conf_thres: float,
                        iou_thres: float,
                        max_det: int = 300,
                        max_nms: int = 30000) -> torch.Tensor:
  candidates = y_pred[..., 4] > conf_thres
  outputs = [y_pred.new_zeros(size=(0, 6)) for _ in range(len(y_pred))]
  for i, x in enumerate(y_pred):
    x = x[candidates[i]]
    if x.size(0) == 0:
      continue

    x = torch.cat([x[:, :5], x[:, 5:] * x[:, 4:5]], 1)

    bounding_boxes = xywh2xyxy(x[:, :4])
    conf, class_indices = x[:, 5:].max(1, keepdim=True)
    x = torch.cat([bounding_boxes, conf, class_indices.float()], 1)[conf.view(-1) > conf_thres]
    if max_nms < x.size(0):
      x = x[x[:, 4].argsort(descending=True)][:max_nms]

    bounding_boxes = x[:, :4]
    scores = x[:, 4]
    indices = torchvision.ops.nms(bounding_boxes, scores, iou_thres)
    if max_det < indices.size(0):
      indices = indices[:max_det]

    outputs[i] = x[indices]

  return outputs


def save_image_tensor(save_path: str, image_tensor: torch.Tensor) -> np.ndarray:
  ext = save_path[save_path.rindex('.'):]
  if ext not in ['.bmp', '.png', '.jpg', '.jpeg']:
    raise ValueError(f'The unsupported image extension is passed (ext={ext}). An image extension must be one of bmp, png, jpg and jpeg.')

  if image_tensor.dtype != torch.uint8:
    raise ValueError(f'Only tensor of which dtype is byte(uint8) can be saved ({image_tensor.dtype} != torch.uint8).')

  image_tensor = image_tensor.cpu().squeeze_()
  if (image_tensor.ndim != 3) or (3 not in [image_tensor.size(0), image_tensor.size(-1)]):
      raise ValueError('The passed tensor has invalid shape.')

  if image_tensor.size(0) == 3:  # assuming that tensor shape is CxHxW.
    image_tensor = image_tensor.permute(1, 2, 0)

  img = cv2.cvtColor(image_tensor.numpy(), cv2.COLOR_RGB2BGR)
  cv2.imwrite(save_path, img)
  return img
