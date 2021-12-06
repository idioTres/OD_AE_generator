import io

import cv2
import torch
import numpy as np
import pycurl

from .. import util
from ..od_attack import YOLOv5VanishAttack


def _prepare_resources():
  yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=False).eval()

  img_url, buffer = 'https://ultralytics.com/images/zidane.jpg', io.BytesIO()
  curl = pycurl.Curl()
  curl.setopt(curl.URL, img_url)
  curl.setopt(curl.FOLLOWLOCATION, True)
  curl.setopt(curl.WRITEDATA, buffer)
  curl.perform()
  curl.close()

  img = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
  img = cv2.imdecode(img, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = util.mksquare(img, size=(640, 640))

  return yolov5, img


_model, _img = _prepare_resources()


def test_yolov5_vanish_attack():
  hp = {'conf_thres': 0.15,
        'iou_thres': 0.8,
        'alpha': 0.002,
        'eps': 0.04,
        'max_iter': 100}

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  attacker = YOLOv5VanishAttack(_model, **hp).to(device)

  img = torch.from_numpy(_img).to(device)
  img = img.permute(2, 0, 1) / 255
  adv_example = attacker(img)

  y_pred = _model(adv_example.unsqueeze(0))[0]
  y_pred = util.non_max_suppression(y_pred, hp['conf_thres'], hp['iou_thres'])[0]
  assert y_pred.size(0) == 0
