def xywh2xyxy(xywh):
  xyxy = xywh.clone().detach_()
  xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
  xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
  xyxy[..., 2] = xywh[..., 2] + xywh[..., 2] / 2
  xyxy[..., 3] = xywh[..., 3] + xywh[..., 3] / 2
  return xyxy
