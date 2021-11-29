import torch
import torchvision


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
